"""
Retrieval-augmented Gemini downstream translator (Gators-style).

For each Spanish caption, retrieve the top-k most similar Spanish->Indigenous
training pairs (BM25 over the AmericasNLP 2021 parallel corpus, TRAIN only --
no dev leakage), put them in the prompt as translation examples, and have
Gemini translate. Pure API; no GPU.

Compares against the fixed NLLB translator on the SAME Spanish input.

Usage:
  OPENROUTER_API_KEY=$(cat ~/.openrouter_key) \
    python baseline/gemini_rag_downstream.py --k 20 --model google/gemini-3-flash-preview
"""
import os, re, json, time, argparse
import requests
from rank_bm25 import BM25Okapi
from sacrebleu.metrics import CHRF

ISO = {"wixarika": "hch", "bribri": "bzd", "guarani": "gn", "nahuatl": "nah"}
LANG_NAME = {"wixarika": "Wixárika (Huichol)", "bribri": "Bribri",
             "guarani": "Guaraní", "nahuatl": "Nahuatl"}
# NLLB downstream on the SAME LLaVA p0_long Spanish input (our baseline to beat):
NLLB_REF = {"wixarika": 16.80, "bribri": 5.85, "guarani": 13.90, "nahuatl": 16.34}
# Gators winner TEST-set scores (for context; dev vs test not directly comparable):
GATORS_TEST = {"wixarika": 17.58, "bribri": 17.90, "guarani": 23.10, "nahuatl": 25.42}
KEY = os.environ["OPENROUTER_API_KEY"]
CHRF = CHRF(word_order=2)


def tok(s):
    return re.findall(r"\w+", s.lower())


def load_corpus(language):
    d = f"data/parallel/{language}"
    es = [l.rstrip("\n") for l in open(f"{d}/train.es", encoding="utf-8")]
    ind = [l.rstrip("\n") for l in open(f"{d}/train.{ISO[language]}", encoding="utf-8")]
    pairs = [(a, b) for a, b in zip(es, ind) if a.strip() and b.strip()]
    return pairs


def translate(es, language, examples, model, retries=4):
    name = LANG_NAME[language]
    instr = (f"You are an expert translator into {name}, an Indigenous language of the Americas. "
             f"Translate the Spanish sentence into {name}. Use the retrieved translation examples "
             f"as a guide for vocabulary, morphology, and spelling. Output only the {name} "
             f"translation, nothing else.")
    ex = "\n".join(f"Spanish: {a}\n{name}: {b}" for a, b in examples)
    prompt = f"{instr}\n\nExamples:\n{ex}\n\nNow translate:\nSpanish: {es}\n{name}:"
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


def run(language, model, k, in_tag):
    pairs = load_corpus(language)
    bm25 = BM25Okapi([tok(a) for a, _ in pairs])
    rows = [json.loads(l) for l in open(
        f"results/spanish_pivots_llava7b/{language}_p0_long_literal_{in_tag}.jsonl")]
    t = time.time()
    preds, refs = [], []
    for i, row in enumerate(rows):
        es = row["generated_caption"]
        scores = bm25.get_scores(tok(es))
        top = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:k]
        examples = [pairs[j] for j in reversed(top)]  # most-relevant last (closest to query)
        preds.append(translate(es, language, examples, model))
        refs.append(row["target_caption"])
        if (i + 1) % 10 == 0:
            print(f"  {language}: {i+1}/{len(rows)}", flush=True)
    vals = [CHRF.sentence_score(p, [r]).score for p, r in zip(preds, refs)]
    gem = round(sum(vals) / len(vals), 2)
    os.makedirs("results/gemini_downstream", exist_ok=True)
    mtag = model.split("/")[-1]
    with open(f"results/gemini_downstream/{language}_{mtag}_rag{k}_translated.txt", "w") as f:
        f.write("\n".join(preds) + "\n")
    return {"language": language, "k": k, "corpus": len(pairs), "gemini_rag": gem,
            "nllb": NLLB_REF[language], "gators_test": GATORS_TEST[language],
            "seconds": round(time.time() - t)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", default="wixarika,bribri,guarani,nahuatl")
    ap.add_argument("--model", default="google/gemini-3-flash-preview")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--in-tag", default="llava-onevision-qwen2-7b-si-hf")
    args = ap.parse_args()

    results = []
    for lang in args.languages.split(","):
        lang = lang.strip()
        print(f"\n=== {lang} | RAG k={args.k} | {args.model} ===")
        results.append(run(lang, args.model, args.k, args.in_tag))

    print("\n=========== RETRIEVAL-AUGMENTED GEMINI vs NLLB (same LLaVA Spanish) ===========")
    print(f"{'language':10s} {'NLLB':>7s} {'Gem+RAG':>9s} {'Δ':>8s} {'(Gators test)':>14s}")
    for r in results:
        d = f"{r['gemini_rag']-r['nllb']:+.2f}"
        print(f"{r['language']:10s} {r['nllb']:>7.2f} {r['gemini_rag']:>9.2f} {d:>8s} "
              f"{r['gators_test']:>14.2f}")
    with open("results/gemini_downstream/SUMMARY_rag.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/gemini_downstream/SUMMARY_rag.json")


if __name__ == "__main__":
    main()
