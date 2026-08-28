"""
Gemini-as-downstream-translator experiment (Gators-style).

Replaces the fixed NLLB translator with Gemini doing Spanish -> Indigenous
translation via few-shot in-context prompting (pilot parallel pairs as
examples). Pure OpenRouter API -- no GPU / Modal needed.

For a controlled comparison we feed the SAME Spanish captions that were
translated by NLLB, so the only thing that changes is the downstream
translator.

Usage:
  OPENROUTER_API_KEY=$(cat ~/.openrouter_key) python baseline/gemini_downstream.py
  ... --model google/gemini-3-flash-preview --languages wixarika,bribri
"""
import os, sys, json, time, argparse
import requests
from sacrebleu.metrics import CHRF

LANG_NAME = {
    "wixarika": "Wixárika (Huichol)",
    "bribri": "Bribri",
    "guarani": "Guaraní",
    "nahuatl": "Nahuatl",
}
# NLLB downstream scores on the SAME input (LLaVA-7B p0_long_literal Spanish).
NLLB_REF = {"wixarika": 16.80, "bribri": 5.85, "guarani": 13.90, "nahuatl": 16.34}
KEY = os.environ["OPENROUTER_API_KEY"]
CHRF = CHRF(word_order=2)


def few_shot_block(language, n):
    pairs = [json.loads(l) for l in open(f"data/pilot/{language}.jsonl")]
    pairs = [p for p in pairs if p.get("spanish_caption") and p.get("target_caption")][:n]
    name = LANG_NAME[language]
    return pairs, "\n".join(f"Spanish: {p['spanish_caption']}\n{name}: {p['target_caption']}" for p in pairs)


def translate(es, language, examples, model, retries=4):
    name = LANG_NAME[language]
    instr = (f"You are an expert translator into {name}, an Indigenous language of the Americas. "
             f"Translate the Spanish sentence into {name}. Use the examples as a guide for spelling, "
             f"morphology, and style. Output only the {name} translation, nothing else.")
    prompt = f"{instr}\n\nExamples:\n{examples}\n\nNow translate:\nSpanish: {es}\n{name}:"
    for a in range(retries):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {KEY}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 120}, timeout=90)
            j = r.json()
            if "choices" in j and j["choices"]:
                return " ".join(j["choices"][0]["message"]["content"].split())
        except Exception:
            pass
        time.sleep(2 * (a + 1))
    return ""


def run(language, model, n_ex, in_tag):
    infile = f"results/spanish_pivots_llava7b/{language}_p0_long_literal_{in_tag}.jsonl"
    rows = [json.loads(l) for l in open(infile)]
    pairs, examples = few_shot_block(language, n_ex)
    t = time.time()
    preds, refs = [], []
    for i, row in enumerate(rows):
        es = row["generated_caption"]
        preds.append(translate(es, language, examples, model))
        refs.append(row["target_caption"])
        if (i + 1) % 10 == 0:
            print(f"  {language}: {i+1}/{len(rows)}", flush=True)
    scores = [CHRF.sentence_score(p, [r]).score for p, r in zip(preds, refs)]
    gem = round(sum(scores) / len(scores), 2)
    # save translations
    os.makedirs("results/gemini_downstream", exist_ok=True)
    mtag = model.split("/")[-1]
    with open(f"results/gemini_downstream/{language}_{mtag}_translated.txt", "w") as f:
        f.write("\n".join(preds) + "\n")
    return {"language": language, "n_examples": len(pairs), "gemini": gem,
            "nllb": NLLB_REF.get(language), "seconds": round(time.time() - t)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", default="wixarika,bribri,guarani,nahuatl")
    ap.add_argument("--model", default="google/gemini-2.5-flash")
    ap.add_argument("--n-examples", type=int, default=20)
    ap.add_argument("--in-tag", default="llava-onevision-qwen2-7b-si-hf")
    args = ap.parse_args()

    results = []
    for lang in args.languages.split(","):
        lang = lang.strip()
        print(f"\n=== {lang} | downstream = {args.model} ===")
        results.append(run(lang, args.model, args.n_examples, args.in_tag))

    print("\n================ DOWNSTREAM: NLLB vs Gemini (same LLaVA Spanish input) ================")
    print(f"{'language':10s} {'NLLB':>7s} {'Gemini':>8s} {'Δ':>8s}")
    for r in results:
        d = f"{r['gemini']-r['nllb']:+.2f}" if r['nllb'] is not None else "—"
        print(f"{r['language']:10s} {r['nllb']:>7.2f} {r['gemini']:>8.2f} {d:>8s}")
    with open("results/gemini_downstream/SUMMARY.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/gemini_downstream/SUMMARY.json")


if __name__ == "__main__":
    main()
