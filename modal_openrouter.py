"""
Modal app: caption AmericasNLP2026 images with hosted frontier VLMs via OpenRouter,
then run the team's Spanish-pivot pipeline (Sheffield NLLB) → ChrF++.

Question: does a much stronger / different captioner break the frozen-MT ceiling
(~19.4 on Wixárika that Qwen-2B/8B/32B all plateau at)?

Prompt = p0_long_literal (the team's best). Each (model × language) runs in its own
container in parallel. Captions via OpenRouter HTTP; translate via NLLB on the volume.

The OpenRouter key is read locally from ~/.openrouter_key and injected as a Modal
Secret — it is NEVER written into this file or the repo.

Usage (from repo root, after `modal setup`):
  modal run baseline/modal_openrouter.py
  modal run baseline/modal_openrouter.py --models qwen3-vl-235b --languages wixarika
  modal volume get americasnlp-out / ./modal_output
"""
import os
import json
import modal

REPO_LOCAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_SIDE = 672
PROMPT_MODE = "p0_long_literal"
PROMPT_TEXT = """Describe this image in Spanish using one or two literal sentences.

Rules:
- Only describe what is visible.
- Include important visible details: people, objects, animals, actions, clothing, place, and background.
- Do not add cultural interpretation unless it is directly visible.
- Do not mention rituals, symbolism, or history.
- Maximum 45 words.
- Output only the Spanish caption."""

# OpenRouter model id keyed by a short tag used in filenames.
MODEL_IDS = {
    "qwen3-vl-32b": "qwen/qwen3-vl-32b-instruct",
    "qwen3-vl-235b": "qwen/qwen3-vl-235b-a22b-instruct",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
}
# MT-supported targets only (Maya yua_Latn is unsupported by submission_3.pt).
TGT_CODE = {
    "wixarika": "hch_Latn",
    "bribri": "bzd_Latn",
    "guarani": "grn_Latn",
    "nahuatl": "nah_Latn",
}
ALL_LANGUAGES = list(TGT_CODE.keys())

app = modal.App("americasnlp-openrouter")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("requests", "pillow", "pandas", "sacrebleu", "sentencepiece",
                 "hydra-core==1.3.3", "omegaconf==2.3.1", "numpy")
    .add_local_dir(
        os.path.join(REPO_LOCAL, "baseline", "americasnlp-2023-sheffield"),
        "/root/repo/baseline/americasnlp-2023-sheffield",
        copy=True, ignore=["fairseq/build/*", "*.pyc", "__pycache__"],
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

# Named secret, created once via: modal secret create openrouter-key OPENROUTER_API_KEY=...
# (Read by the container at runtime; never stored in this file or the repo.)
or_secret = modal.Secret.from_name("openrouter-key")

CKPT = "/out/submission_3.pt"


@app.function(image=image, gpu="T4", timeout=60 * 60, volumes={"/out": vol}, secrets=[or_secret])
def run_one(language: str, tag: str):
    import io, time, base64, subprocess
    import requests
    import pandas as pd
    from PIL import Image as PILImage
    from sacrebleu.metrics import CHRF

    os.chdir("/root/repo")
    key = os.environ["OPENROUTER_API_KEY"]
    model = MODEL_IDS[tag]
    tgt = TGT_CODE[language]

    data_dir = f"data/dev/{language}"
    df = pd.read_json(f"{data_dir}/{language}.jsonl", lines=True)
    df["filepath"] = df["filename"].apply(
        lambda x: os.path.join(data_dir, "images", os.path.basename(str(x)))
    )

    def encode(fp):
        im = PILImage.open(fp).convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        buf = io.BytesIO(); im.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()

    def caption(fp, retries=4):
        b64 = encode(fp)
        last = ""
        for a in range(retries):
            try:
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": model, "messages": [{"role": "user", "content": [
                        {"type": "text", "text": PROMPT_TEXT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ]}], "max_tokens": 120}, timeout=120,
                )
                j = r.json()
                if "choices" in j and j["choices"]:
                    return " ".join(j["choices"][0]["message"]["content"].split())
                last = str(j.get("error"))
            except Exception as e:
                last = str(e)
            time.sleep(2 * (a + 1))
        print(f"[{language}/{tag}] caption failed: {last[:160]}", flush=True)
        return ""

    t = time.time()
    caps = [caption(fp) for fp in df["filepath"]]
    caption_s = time.time() - t
    n_ok = sum(1 for c in caps if c.strip())
    print(f"[{language}/{tag}] {n_ok}/{len(df)} captions in {caption_s:.0f}s", flush=True)

    stem = f"/out/{language}_{PROMPT_MODE}_{tag}"
    out = df.copy(); out["generated_caption"] = caps; out["model"] = model
    out.to_json(f"{stem}.jsonl", orient="records", lines=True, force_ascii=False)
    with open(f"{stem}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join((c or ".") for c in caps) + "\n")

    # Translate via NLLB and score ChrF++
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

    result = {
        "language": language, "model": tag, "score": score, "n_ok": n_ok,
        "caption_s": round(caption_s, 1), "translate_s": round(translate_s, 1),
    }
    with open(f"/out/OR_{language}_{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print(f"[{language}/{tag}] score={score} (n_ok={n_ok})", flush=True)
    return result


@app.function(image=image.pip_install("requests"), volumes={"/out": vol})
def _write_csv(content: str):
    with open("/out/RESULTS_openrouter.csv", "w") as f:
        f.write(content)
    vol.commit()


@app.local_entrypoint()
def main(models: str = "qwen3-vl-32b,qwen3-vl-235b,gemini-2.5-flash",
         languages: str = "wixarika,bribri,guarani,nahuatl"):
    tags = [m.strip() for m in models.split(",") if m.strip()]
    langs = [l.strip() for l in languages.split(",") if l.strip()]
    assert all(t in MODEL_IDS for t in tags), f"models must be subset of {list(MODEL_IDS)}"
    assert all(l in ALL_LANGUAGES for l in langs), f"languages must be subset of {ALL_LANGUAGES}"

    pairs = [(l, t) for t in tags for l in langs]
    print(f"Spawning {len(pairs)} (language × model) durable jobs:")
    # .spawn() = fire-and-forget; each job runs server-side to completion and commits
    # OR_<lang>_<tag>.json to the volume even if this local client disconnects.
    calls = [(l, t, run_one.spawn(l, t)) for (l, t) in pairs]
    for l, t, c in calls:
        print(f"  {t:16s} × {l:9s}  {c.object_id}")
    print()

    results = []
    for l, t, c in calls:
        try:
            results.append(c.get())
        except Exception as e:
            print(f"  (live get failed for {t}×{l}: {e}; result is on volume as OR_{l}_{t}.json)")

    # Pivot: model × language ChrF++
    by = {(r["language"], r["model"]): r for r in results}
    print("\n================= ChrF++  (model × language, p0_long_literal) =================")
    print(f"{'model':18s}" + "".join(f"{l:>11s}" for l in langs))
    for t in tags:
        cells = "".join(
            (f"{by[(l, t)]['score']:>11.2f}" if by.get((l, t)) and by[(l, t)]['score'] is not None else f"{'—':>11s}")
            for l in langs
        )
        print(f"{t:18s}{cells}")

    print("\n================= TIMING (caption + translate, s) =================")
    for r in results:
        print(f"{r['model']:16s} × {r['language']:9s}  caption={r['caption_s']:6.0f}  "
              f"translate={r['translate_s']:6.0f}  n_ok={r['n_ok']}/50  score={r['score']}")

    import io
    buf = io.StringIO()
    buf.write("model,language,score,n_ok,caption_s,translate_s\n")
    for r in results:
        buf.write(f"{r['model']},{r['language']},{r['score']},{r['n_ok']},{r['caption_s']},{r['translate_s']}\n")
    _write_csv.remote(buf.getvalue())
    print("\nReference (team, Wixárika p0_long_literal): Qwen-2B 19.36 | 8B 19.11 | 32B 18.51 | llava-7B 16.80")
    print("Saved RESULTS_openrouter.csv to volume.")
