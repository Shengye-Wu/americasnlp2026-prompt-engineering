"""
Modal app: llava-onevision-qwen2-7b-si-hf caption→MT→ChrF++ pipeline for ALL
AmericasNLP2026 languages, in parallel, with per-language timing.

Stages (one container per language, run in parallel via .map):
  captions  — llava generates Spanish captions for all prompt modes (GPU)
  translate — NLLB (Sheffield submission_3) translates + ChrF++ (GPU, NO llava loaded)
  all       — captions then translate, freeing llava before MT to avoid OOM

Why two passes: holding llava (~16 GB) on the GPU while the translate subprocess
loads NLLB OOMs a 24 GB A10G. The translate pass loads NLLB only, so it gets the
whole GPU. Captions persist to the volume, so translate can re-run independently.

Usage (from repo root, after `modal setup`):
  modal run baseline/modal_run.py --stage translate     # re-score using saved captions
  modal run baseline/modal_run.py --stage all           # full pipeline from scratch
  modal run baseline/modal_run.py --stage all --languages wixarika,bribri
  modal volume get americasnlp-out / ./modal_output     # pull outputs locally

Maya (yua_Latn) is not in the MT's langs_extra.txt → only p4_direct_target
(generated directly in-language, no MT) is scored for Maya.
"""
import os
import json
import modal

REPO_LOCAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "llava-hf/llava-onevision-qwen2-7b-si-hf"
MODEL_TAG = MODEL.split("/")[-1]
SPLIT = "dev"
MAX_NEW_TOKENS = 96
MAX_SIDE = 672

ALL_LANGUAGES = ["wixarika", "bribri", "guarani", "nahuatl", "maya"]
TGT_CODE = {
    "wixarika": "hch_Latn",
    "bribri": "bzd_Latn",
    "guarani": "grn_Latn",
    "nahuatl": "nah_Latn",
    "maya": "yua_Latn",  # unsupported by submission_3.pt
}
PROMPT_MODES = [
    "p0_short_literal",
    "p0_medium_literal",
    "p0_long_literal",
    "p1_culture_aware",
    "p2_translation_friendly",
    "p2b_translation_friendly_detailed",
    "p3_object_action",
    "p4_direct_target",  # generated directly in target language; no MT
]
LANGS_FILE = "baseline/americasnlp-2023-sheffield/NLLB-inference/langs_extra.txt"
CKPT = "/out/submission_3.pt"

app = modal.App("americasnlp-llava")

base_pip = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install(
        "torch", "torchvision", "transformers>=4.45", "accelerate",
        "sacrebleu", "sentencepiece", "gdown", "pillow", "pandas", "tqdm", "numpy",
    )
)

pipeline_image = (
    base_pip
    .add_local_dir(
        os.path.join(REPO_LOCAL, "baseline"), "/root/repo/baseline",
        copy=True,
        ignore=["output/*", "*.pyc", "__pycache__", "americasnlp-2023-sheffield/fairseq/build/*"],
    )
    .add_local_dir(
        os.path.join(REPO_LOCAL, "data", "dev"), "/root/repo/data/dev",
        copy=True, ignore=["*.pyc", "__pycache__"],
    )
    .run_commands(
        "cd /root/repo/baseline/americasnlp-2023-sheffield && pip install -r requirements.txt",
        "rm -rf /root/repo/baseline/americasnlp-2023-sheffield/fairseq/build",
        "cd /root/repo/baseline/americasnlp-2023-sheffield && pip install -e fairseq/",
    )
)

vol = modal.Volume.from_name("americasnlp-out", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


# ── Shared helpers (run inside containers) ───────────────────────────────────
def _generate_captions(language):
    """Load llava, caption all prompts for one language, save to volume. Returns caption_s."""
    import sys, time, torch, pandas as pd
    from PIL import Image as PILImage
    from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
    sys.path.insert(0, "/root/repo/baseline")
    from run_prompt_experiment import build_prompt, LANG_INFO, clean_text

    info = LANG_INFO[language]
    data_dir = f"data/dev/{language}"
    df = pd.read_json(f"{data_dir}/{language}.jsonl", lines=True)
    df["filepath"] = df["filename"].apply(
        lambda x: os.path.join(data_dir, "images", os.path.basename(str(x)))
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t = time.time()
    processor = AutoProcessor.from_pretrained(MODEL)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    model_load_s = time.time() - t
    print(f"[{language}] model loaded {model_load_s:.1f}s", flush=True)

    images = []
    for fp in df["filepath"]:
        im = PILImage.open(fp).convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        images.append(im)

    def caption(image, prompt_text):
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True)
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return clean_text(processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))

    caption_s = 0.0
    for mode in PROMPT_MODES:
        prompt = build_prompt(mode, info["culture"], info["language_name"])
        print(f"[{language}] caption {mode}", flush=True)
        t = time.time()
        caps = [caption(img, prompt) for img in images]
        caption_s += time.time() - t
        stem = f"/out/{language}_{mode}_{MODEL_TAG}"
        out_df = df.copy()
        out_df["generated_caption"] = caps
        out_df["prompt_mode"] = mode
        out_df["model"] = MODEL
        out_df.to_json(f"{stem}.jsonl", orient="records", lines=True, force_ascii=False)
        with open(f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(c.replace("\n", " ") for c in caps) + "\n")

    # Free llava before any MT step on the same GPU
    del model, processor
    import gc; gc.collect()
    torch.cuda.empty_cache()
    return round(caption_s, 1), round(model_load_s, 1)


def _translate_and_score(language):
    """Translate saved captions (one NLLB load via concatenation) + ChrF++. No llava loaded."""
    import time, subprocess, pandas as pd
    from sacrebleu.metrics import CHRF

    tgt = TGT_CODE[language]
    supported = set(open(LANGS_FILE).read().strip().split(","))
    mt_ok = tgt in supported
    chrf = CHRF(word_order=2)
    scores, translate_s = {}, 0.0
    mt_modes = [m for m in PROMPT_MODES if m != "p4_direct_target"]

    if mt_ok:
        order, all_lines, counts = [], [], {}
        for m in mt_modes:
            with open(f"/out/{language}_{m}_{MODEL_TAG}.txt") as f:
                lines = [(l.rstrip("\n") or ".") for l in f]
            counts[m] = len(lines); order.append(m); all_lines.extend(lines)
        cin, cout = f"/out/_cat_{language}_in.txt", f"/out/_cat_{language}_out.txt"
        with open(cin, "w") as f:
            f.write("\n".join(all_lines) + "\n")
        t = time.time()
        subprocess.run(
            ["python", "baseline/americasnlp-2023-sheffield/translate.py",
             "--checkpoint", CKPT, "--input", cin, "--output", cout,
             "--src", "spa_Latn", "--tgt", tgt],
            check=True,
        )
        translate_s = time.time() - t
        with open(cout) as f:
            out_lines = [l.rstrip("\n") for l in f]
        if len(out_lines) != len(all_lines):
            print(f"[{language}] MISALIGN {len(out_lines)} vs {len(all_lines)} — marking MT prompts failed", flush=True)
            for m in mt_modes:
                scores[m] = None
        else:
            idx = 0
            for m in order:
                chunk = out_lines[idx:idx + counts[m]]; idx += counts[m]
                stem = f"/out/{language}_{m}_{MODEL_TAG}"
                with open(f"{stem}_translated.txt", "w") as f:
                    f.write("\n".join(chunk) + "\n")
                ref = pd.read_json(f"{stem}.jsonl", lines=True)
                vals = [chrf.sentence_score(p, [r]).score for p, r in zip(chunk, ref["target_caption"])]
                scores[m] = round(sum(vals) / len(vals), 2)
                print(f"[{language}] {m}: {scores[m]}", flush=True)
    else:
        for m in mt_modes:
            scores[m] = None
        print(f"[{language}] MT target {tgt} unsupported — only p4 scored", flush=True)

    # p4: direct in-language generation, no MT
    stem4 = f"/out/{language}_p4_direct_target_{MODEL_TAG}"
    ref4 = pd.read_json(f"{stem4}.jsonl", lines=True)
    with open(f"{stem4}.txt") as f:
        preds4 = [l.rstrip("\n") for l in f]
    if len(preds4) == len(ref4):
        vals = [chrf.sentence_score(p, [r]).score for p, r in zip(preds4, ref4["target_caption"])]
        scores["p4_direct_target"] = round(sum(vals) / len(vals), 2)
    else:
        scores["p4_direct_target"] = None
    print(f"[{language}] p4_direct_target: {scores['p4_direct_target']}", flush=True)

    return scores, round(translate_s, 1), mt_ok


def _finalize(language, scores, caption_s, translate_s, model_load_s, mt_ok):
    best = max((m for m in scores if scores[m] is not None), key=lambda m: scores[m], default=None)
    result = {
        "language": language, "mt_ok": mt_ok,
        "model_load_s": model_load_s, "caption_s": caption_s,
        "translate_s": translate_s,
        "pipeline_s": round((caption_s or 0) + (translate_s or 0), 1),
        "scores": scores, "best_prompt": best,
        "best_score": scores.get(best) if best else None,
    }
    with open(f"/out/PIPELINE_{language}.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print(f"[{language}] DONE best={best}={result['best_score']} translate={translate_s}s", flush=True)
    return result


# ── Modal functions ──────────────────────────────────────────────────────────
@app.function(image=pipeline_image, gpu="A10G", timeout=2 * 60 * 60,
              volumes={"/out": vol, "/root/.cache/huggingface": hf_cache})
def run_all(language: str):
    os.chdir("/root/repo")
    assert os.path.exists(CKPT), "submission_3.pt missing from volume"
    caption_s, model_load_s = _generate_captions(language)
    scores, translate_s, mt_ok = _translate_and_score(language)
    return _finalize(language, scores, caption_s, translate_s, model_load_s, mt_ok)


@app.function(image=pipeline_image, gpu="A10G", timeout=60 * 60, volumes={"/out": vol})
def run_translate(language: str):
    """Re-score from saved captions (no llava). Pulls caption_s from prior PIPELINE json if present."""
    os.chdir("/root/repo")
    assert os.path.exists(CKPT), "submission_3.pt missing from volume"
    prior = f"/out/PIPELINE_{language}.json"
    caption_s = model_load_s = None
    if os.path.exists(prior):
        p = json.load(open(prior))
        caption_s, model_load_s = p.get("caption_s"), p.get("model_load_s")
    scores, translate_s, mt_ok = _translate_and_score(language)
    return _finalize(language, scores, caption_s, translate_s, model_load_s, mt_ok)


@app.function(image=base_pip, volumes={"/out": vol})
def _write_csv(content: str):
    with open("/out/RESULTS_all_languages.csv", "w") as f:
        f.write(content)
    vol.commit()


@app.local_entrypoint()
def main(stage: str = "all", languages: str = ""):
    langs = [l.strip() for l in languages.split(",") if l.strip()] or ALL_LANGUAGES
    bad = [l for l in langs if l not in ALL_LANGUAGES]
    if bad:
        raise ValueError(f"Unknown language(s): {bad}")
    fn = {"all": run_all, "translate": run_translate}.get(stage)
    if fn is None:
        raise ValueError("stage must be 'all' or 'translate'")

    print(f"Stage='{stage}' for {langs} (parallel)\n")
    results = list(fn.map(langs, order_outputs=True))

    print("\n================= PER-LANGUAGE TIMING =================")
    print(f"{'language':10s} {'caption(s)':>11s} {'translate(s)':>13s} {'pipeline(s)':>12s}")
    for r in results:
        cs = r['caption_s'] if r['caption_s'] is not None else 0.0
        print(f"{r['language']:10s} {cs:11.1f} {r['translate_s']:13.1f} {r['pipeline_s']:12.1f}")

    print("\n================= ChrF++ (language × prompt) =================")
    print("language   " + " ".join(f"{m:>9s}"[:9] for m in
          ['short','medium','long','p1cult','p2mt','p2bmt','p3obj','p4dir']))
    for r in results:
        row = " ".join((f"{r['scores'].get(m):>9.2f}" if r['scores'].get(m) is not None else f"{'—':>9s}")
                       for m in PROMPT_MODES)
        print(f"{r['language']:10s} {row}" + ("" if r['mt_ok'] else "   (MT unsupported)"))

    print("\n================= BEST PER LANGUAGE =================")
    for r in results:
        print(f"{r['language']:10s} {r['best_prompt']} = {r['best_score']}")

    import io
    buf = io.StringIO()
    buf.write("language,mt_ok,caption_s,translate_s,pipeline_s," + ",".join(PROMPT_MODES) + ",best_prompt,best_score\n")
    for r in results:
        cells = [r["language"], r["mt_ok"], r["caption_s"], r["translate_s"], r["pipeline_s"]]
        cells += [(r["scores"].get(m) if r["scores"].get(m) is not None else "") for m in PROMPT_MODES]
        cells += [r["best_prompt"], r["best_score"]]
        buf.write(",".join(str(c) for c in cells) + "\n")
    _write_csv.remote(buf.getvalue())
    print("\nSaved RESULTS_all_languages.csv to volume. Pull: modal volume get americasnlp-out / ./modal_output")
