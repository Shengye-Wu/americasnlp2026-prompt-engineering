"""
Modal app: FEW-SHOT captioning experiment.


Pipeline per (model x language):
  20 pilot examples + query dev image  ->  OpenRouter VLM  ->  Spanish caption
  ->  NLLB MT (Sheffield submission_3) ->  Wixárika  ->  ChrF++ (50 dev images)

Usage (from repo root, after `modal setup`):
  modal run baseline/modal/modal_fewshot.py
  modal run baseline/modal/modal_fewshot.py --models gemini-3-flash --languages wixarika
  modal run baseline/modal/modal_fewshot.py --shots 10
  modal volume get americasnlp-out / ./modal_output

Key is a named Modal secret (openrouter-key); never written into this file.
"""
import os
import json
import modal

REPO_LOCAL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N_SHOTS_DEFAULT = 20
EX_SIDE = 448   # downscale example images (cheaper tokens)
Q_SIDE = 672    # query image

INSTR = (
    "You are writing Spanish captions for images. Study the example image and caption "
    "pairs below, then caption the FINAL image in the same style: one or two literal "
    "sentences, describe only what is visible, no invented cultural or symbolic meaning, "
    "maximum 45 words. Output only the Spanish caption."
)

MODEL_IDS = {
    "qwen3-vl-32b": "qwen/qwen3-vl-32b-instruct",
    "qwen3-vl-235b": "qwen/qwen3-vl-235b-a22b-instruct",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-3-flash": "google/gemini-3-flash-preview",
}
TGT_CODE = {"wixarika": "hch_Latn", "bribri": "bzd_Latn", "guarani": "grn_Latn", "nahuatl": "nah_Latn"}
ALL_LANGUAGES = list(TGT_CODE.keys())
CKPT = "/out/submission_3.pt"

app = modal.App("americasnlp-fewshot")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("requests", "pillow", "pandas", "sacrebleu", "sentencepiece",
                 "hydra-core==1.3.3", "omegaconf==2.3.1", "numpy")
    .add_local_dir(os.path.join(REPO_LOCAL, "baseline", "americasnlp-2023-sheffield"),
                   "/root/repo/baseline/americasnlp-2023-sheffield",
                   copy=True, ignore=["fairseq/build/*", "*.pyc", "__pycache__"])
    .add_local_dir(os.path.join(REPO_LOCAL, "data", "dev"), "/root/repo/data/dev",
                   copy=True, ignore=["*.pyc", "__pycache__"])
    .add_local_dir(os.path.join(REPO_LOCAL, "data", "pilot"), "/root/repo/data/pilot",
                   copy=True, ignore=["*.pyc", "__pycache__"])
    .run_commands(
        "cd /root/repo/baseline/americasnlp-2023-sheffield && pip install -r requirements.txt",
        "rm -rf /root/repo/baseline/americasnlp-2023-sheffield/fairseq/build",
        "cd /root/repo/baseline/americasnlp-2023-sheffield && pip install -e fairseq/",
    )
)

vol = modal.Volume.from_name("americasnlp-out", create_if_missing=True)
or_secret = modal.Secret.from_name("openrouter-key")


@app.function(image=image, gpu="T4", timeout=60 * 60, volumes={"/out": vol}, secrets=[or_secret])
def run_fewshot(language: str, tag: str, shots: int = N_SHOTS_DEFAULT):
    import io, time, base64, subprocess
    import requests
    import pandas as pd
    from PIL import Image as PILImage
    from sacrebleu.metrics import CHRF

    os.chdir("/root/repo")
    key = os.environ["OPENROUTER_API_KEY"]
    model = MODEL_IDS[tag]
    tgt = TGT_CODE[language]

    def enc(path, side):
        im = PILImage.open(path).convert("RGB")
        im.thumbnail((side, side))
        b = io.BytesIO(); im.save(b, format="JPEG")
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

    # Build the shared few-shot block from the pilot set (image + reference Spanish caption)
    pilot = [json.loads(l) for l in open(f"data/pilot/{language}.jsonl")][:shots]
    example_content = [{"type": "text", "text": INSTR + "\n\nExamples:"}]
    for i, ex in enumerate(pilot, 1):
        p = os.path.join(f"data/pilot/images/{language}", os.path.basename(ex["filename"]))
        example_content.append({"type": "image_url", "image_url": {"url": enc(p, EX_SIDE)}})
        example_content.append({"type": "text", "text": f"Caption {i}: {ex['spanish_caption']}"})
    example_content.append({"type": "text", "text": "Now write the Spanish caption for this image:"})
    print(f"[{language}/{tag}] built {len(pilot)}-shot prompt", flush=True)

    # Dev images (the evaluation set)
    data_dir = f"data/dev/{language}"
    df = pd.read_json(f"{data_dir}/{language}.jsonl", lines=True)
    df["filepath"] = df["filename"].apply(
        lambda x: os.path.join(data_dir, "images", os.path.basename(str(x)))
    )

    def caption(fp, retries=4):
        content = example_content + [{"type": "image_url", "image_url": {"url": enc(fp, Q_SIDE)}}]
        last = ""
        for a in range(retries):
            try:
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": model, "messages": [{"role": "user", "content": content}],
                          "max_tokens": 120}, timeout=180,
                )
                j = r.json()
                if "choices" in j and j["choices"]:
                    return " ".join(j["choices"][0]["message"]["content"].split())
                last = str(j.get("error"))
            except Exception as e:
                last = str(e)
            time.sleep(3 * (a + 1))
        print(f"[{language}/{tag}] caption failed: {last[:150]}", flush=True)
        return ""

    t = time.time()
    caps = [caption(fp) for fp in df["filepath"]]
    caption_s = time.time() - t
    n_ok = sum(1 for c in caps if c.strip())
    print(f"[{language}/{tag}] {n_ok}/{len(df)} captions in {caption_s:.0f}s", flush=True)

    stem = f"/out/{language}_fewshot{shots}_{tag}"
    out = df.copy(); out["generated_caption"] = caps; out["model"] = model
    out.to_json(f"{stem}.jsonl", orient="records", lines=True, force_ascii=False)
    with open(f"{stem}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join((c or ".") for c in caps) + "\n")

    # Translate + score
    cin, cout = f"{stem}_src.txt", f"{stem}_translated.txt"
    with open(cin, "w", encoding="utf-8") as f:
        f.write("\n".join((c or ".") for c in caps) + "\n")
    t = time.time()
    subprocess.run(
        ["python", "baseline/americasnlp-2023-sheffield/translate.py",
         "--checkpoint", CKPT, "--input", cin, "--output", cout,
         "--src", "spa_Latn", "--tgt", tgt], check=True,
    )
    translate_s = time.time() - t
    with open(cout) as f:
        preds = [l.rstrip("\n") for l in f]

    chrf = CHRF(word_order=2)
    score = None
    if len(preds) == len(df):
        vals = [chrf.sentence_score(p, [r]).score for p, r in zip(preds, df["target_caption"])]
        score = round(sum(vals) / len(vals), 2)

    result = {"language": language, "model": tag, "shots": shots, "score": score,
              "n_ok": n_ok, "caption_s": round(caption_s, 1), "translate_s": round(translate_s, 1)}
    with open(f"/out/FEWSHOT_{language}_{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print(f"[{language}/{tag}] FEW-SHOT score={score} (n_ok={n_ok})", flush=True)
    return result


@app.local_entrypoint()
def main(models: str = "qwen3-vl-32b,qwen3-vl-235b,gemini-2.5-flash,gemini-3-flash",
         languages: str = "wixarika", shots: int = N_SHOTS_DEFAULT):
    tags = [m.strip() for m in models.split(",") if m.strip()]
    langs = [l.strip() for l in languages.split(",") if l.strip()]
    assert all(t in MODEL_IDS for t in tags), f"models subset of {list(MODEL_IDS)}"
    assert all(l in ALL_LANGUAGES for l in langs), f"languages subset of {ALL_LANGUAGES}"

    pairs = [(l, t) for t in tags for l in langs]
    print(f"Spawning {len(pairs)} few-shot ({shots}-shot) jobs:")
    calls = [(l, t, run_fewshot.spawn(l, t, shots)) for (l, t) in pairs]
    for l, t, c in calls:
        print(f"  {t:16s} × {l:9s}  {c.object_id}")

    results = []
    for l, t, c in calls:
        try:
            results.append(c.get())
        except Exception as e:
            print(f"  (live get failed for {t}×{l}: {e}; result on volume as FEWSHOT_{l}_{t}.json)")

    # Zero-shot reference (same models/prompt = p0_long_literal, from earlier runs)
    zshot = {"qwen3-vl-32b": 18.19, "qwen3-vl-235b": 18.42, "gemini-2.5-flash": 18.74, "gemini-3-flash": None}
    print("\n============ FEW-SHOT vs ZERO-SHOT (Wixárika ChrF++) ============")
    print(f"{'model':18s}{'few-shot':>10s}{'zero-shot':>11s}{'Δ':>8s}")
    for r in results:
        z = zshot.get(r["model"])
        d = (f"{r['score']-z:+.2f}" if (z is not None and r['score'] is not None) else "—")
        zs = (f"{z:.2f}" if z is not None else "—")
        fs = (f"{r['score']:.2f}" if r['score'] is not None else "fail")
        print(f"{r['model']:18s}{fs:>10s}{zs:>11s}{d:>8s}")
    print("\nRef best zero-shot overall: Qwen-2B 19.36 (Hamza). Ceiling to beat ~19.4.")

    import io
    buf = io.StringIO()
    buf.write("model,language,shots,score,n_ok,caption_s,translate_s\n")
    for r in results:
        buf.write(f"{r['model']},{r['language']},{r['shots']},{r['score']},{r['n_ok']},{r['caption_s']},{r['translate_s']}\n")
    _write_csv.remote(buf.getvalue())
    print("Saved RESULTS_fewshot.csv to volume.")


@app.function(image=modal.Image.debian_slim(), volumes={"/out": vol})
def _write_csv(content: str):
    with open("/out/RESULTS_fewshot.csv", "w") as f:
        f.write(content)
    vol.commit()
