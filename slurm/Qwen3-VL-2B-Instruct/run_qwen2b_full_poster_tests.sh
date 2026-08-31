#!/bin/bash
set -euo pipefail

# ============================================================
# Qwen3-VL-2B full poster test sweep
# AmericasNLP 2026 Cultural Image Captioning
#
# What this script does:
# 1) Runs every prompt mode currently available in scripts/run_qwen_prompt_experiment.py
#    for the four poster languages: Wixárika, Bribri, Guaraní, Nahuatl.
# 2) Uses the same workflow for Spanish-pivot prompts:
#       Qwen3-VL-2B -> Spanish caption -> Sheffield MT -> target language -> ChrF++
# 3) Evaluates p4_direct_target directly without Spanish->MT.
# 4) Creates poster-ready CSV summaries.
# 5) Creates one tar.gz archive with all Qwen2B poster results.
#
# Run from LRZ project folder with:
#   cd ~/americasnlp2026
#   bash run_qwen2b_full_poster_tests.sh
# ============================================================

chmod +x "$0" 2>/dev/null || true

cd ~/americasnlp2026

DSS="<LRZ_DSS_PATH>"
export HF_HOME="$DSS/<LRZ_ACCOUNT>/hf_cache"
export HF_HUB_CACHE="$DSS/<LRZ_ACCOUNT>/hf_cache/hub"
export TRANSFORMERS_CACHE="$DSS/<LRZ_ACCOUNT>/hf_cache"

MODEL="Qwen/Qwen3-VL-2B-Instruct"
MODEL_TAG="qwen3_vl_2b"
SAMPLES=50
SPLIT="dev"

mkdir -p baseline/output logs

# Clean previous Qwen2B poster sweep outputs so this run starts fresh.
rm -f baseline/output/lrz_dev_*_2b_*_50*
rm -f baseline/output/qwen2b_poster_*.csv
rm -f lrz_dev_qwen2b_full_poster_results.tar.gz

# Guaraní filename/path fix, needed because its JSONL filenames include data/guarani/images/...
mkdir -p data/dev/guarani/data/guarani
rm -rf data/dev/guarani/data/guarani/images
ln -s ../../images data/dev/guarani/data/guarani/images || true

SUMMARY="baseline/output/qwen2b_poster_complete_summary.csv"
cat > "$SUMMARY" <<'CSV'
model,language,split,samples,prompt,workflow,target_code,final_chrf,status,caption_file,translation_file,score_file
CSV

run_test () {
  local LANG="$1"
  local TGT="$2"
  local PROMPT="$3"

  echo ""
  echo "============================================================"
  echo "Running: language=${LANG} | target=${TGT} | prompt=${PROMPT}"
  echo "============================================================"

  deactivate 2>/dev/null || true

  local OUT="baseline/output/lrz_dev_${LANG}_2b_${PROMPT}_50"
  local CAPTION_FILE="${OUT}.txt"
  local JSONL_FILE="${OUT}.jsonl"
  local TRANSLATION_FILE="${OUT}_translated_${LANG}_50.txt"
  local SCORE_FILE="${OUT}_final_score.txt"

  rm -f "${OUT}"* || true

  python baseline/captioning/run_qwen_prompt_experiment.py \
    --model "$MODEL" \
    --language "$LANG" \
    --split "$SPLIT" \
    --prompt-mode "$PROMPT" \
    --max-samples "$SAMPLES" \
    --max-new-tokens 96 \
    --output-prefix "$OUT"

  echo "Caption line count:"
  wc -l "$CAPTION_FILE"
  echo "First 3 generated captions:"
  head -n 3 "$CAPTION_FILE"

  if [ "$PROMPT" = "p4_direct_target" ]; then
    echo "Direct target-language generation: evaluating generated target captions directly."
    python baseline/eval.py \
      --dataframe "$JSONL_FILE" \
      --translations "$CAPTION_FILE" \
      | tee "$SCORE_FILE"

    local SCORE
    SCORE=$(grep "Mean chrF++ score:" "$SCORE_FILE" | awk -F': ' '{print $2}')
    echo "$MODEL,$LANG,$SPLIT,$SAMPLES,$PROMPT,direct_target_generation,$TGT,$SCORE,completed,$CAPTION_FILE,$CAPTION_FILE,$SCORE_FILE" >> "$SUMMARY"
  else
    echo "Spanish-pivot generation: translating Spanish captions to target language."
    source mt_env/bin/activate

    CUDA_VISIBLE_DEVICES="" python baseline/americasnlp-2023-sheffield/translate.py \
      --checkpoint submission_3.pt \
      --input "$CAPTION_FILE" \
      --output "$TRANSLATION_FILE" \
      --src spa_Latn \
      --tgt "$TGT"

    echo "Translation line count:"
    wc -l "$TRANSLATION_FILE"

    python baseline/eval.py \
      --dataframe "$JSONL_FILE" \
      --translations "$TRANSLATION_FILE" \
      | tee "$SCORE_FILE"

    local SCORE
    SCORE=$(grep "Mean chrF++ score:" "$SCORE_FILE" | awk -F': ' '{print $2}')
    echo "$MODEL,$LANG,$SPLIT,$SAMPLES,$PROMPT,Spanish->Sheffield MT,$TGT,$SCORE,completed,$CAPTION_FILE,$TRANSLATION_FILE,$SCORE_FILE" >> "$SUMMARY"

    deactivate 2>/dev/null || true
  fi

  echo "Finished: ${LANG} | ${PROMPT}"
}

# All prompt modes currently available in scripts/run_qwen_prompt_experiment.py.
# These cover the full poster logic: caption length, prompt strategy, direct target failure, and extra prompt ablations.
PROMPTS=(
  p0_simple
  p0_short_literal
  p0_medium_literal
  p0_long_literal
  p1_culture_aware
  p2_translation_friendly
  p2b_translation_friendly_detailed
  p3_object_action
  p4_direct_target
)

# Four poster languages.
LANGS=(wixarika bribri guarani nahuatl)
TGTS=(hch_Latn bzd_Latn grn_Latn nah_Latn)

for i in "${!LANGS[@]}"; do
  LANG="${LANGS[$i]}"
  TGT="${TGTS[$i]}"
  for PROMPT in "${PROMPTS[@]}"; do
    run_test "$LANG" "$TGT" "$PROMPT"
  done
done

echo ""
echo "============================================================"
echo "Creating poster-ready summary CSV files"
echo "============================================================"

python - <<'PY'
from pathlib import Path
import pandas as pd

summary_path = Path("baseline/output/qwen2b_poster_complete_summary.csv")
df = pd.read_csv(summary_path)
df["final_chrf"] = pd.to_numeric(df["final_chrf"], errors="coerce")

# 1) Full summary sorted for inspection
df_sorted = df.sort_values(["language", "prompt"])
df_sorted.to_csv("baseline/output/qwen2b_poster_complete_summary_sorted.csv", index=False)

# 2) Caption length effect: short vs medium vs long
length_prompts = ["p0_short_literal", "p0_medium_literal", "p0_long_literal"]
length = df[df["prompt"].isin(length_prompts)].pivot(index="language", columns="prompt", values="final_chrf")
length = length.reindex(columns=length_prompts)
length.reset_index().to_csv("baseline/output/qwen2b_caption_length_effect.csv", index=False)

# 3) Prompt strategy face-off for poster
strategy_prompts = [
    "p0_simple",
    "p0_long_literal",
    "p1_culture_aware",
    "p2_translation_friendly",
    "p2b_translation_friendly_detailed",
    "p3_object_action",
]
strategy = df[df["prompt"].isin(strategy_prompts)].pivot(index="language", columns="prompt", values="final_chrf")
strategy = strategy.reindex(columns=strategy_prompts)
strategy.reset_index().to_csv("baseline/output/qwen2b_prompt_strategy_faceoff.csv", index=False)

# 4) Direct target failure: best Spanish pivot vs p4 direct
rows = []
for lang, group in df.groupby("language"):
    pivot = group[group["workflow"] == "Spanish->Sheffield MT"]
    direct = group[group["prompt"] == "p4_direct_target"]
    best_pivot_row = pivot.loc[pivot["final_chrf"].idxmax()]
    direct_score = direct["final_chrf"].iloc[0] if len(direct) else None
    ratio = best_pivot_row["final_chrf"] / direct_score if direct_score and direct_score > 0 else None
    rows.append({
        "language": lang,
        "best_spanish_pivot_prompt": best_pivot_row["prompt"],
        "best_spanish_pivot_chrf": round(float(best_pivot_row["final_chrf"]), 2),
        "p4_direct_target_chrf": round(float(direct_score), 2) if direct_score is not None else None,
        "pivot_vs_direct_ratio": round(float(ratio), 2) if ratio is not None else None,
    })
pd.DataFrame(rows).sort_values("language").to_csv("baseline/output/qwen2b_direct_target_failure.csv", index=False)

# 5) Best prompt per language
best = df.loc[df.groupby("language")["final_chrf"].idxmax()].copy()
best = best[["language", "prompt", "workflow", "final_chrf"]].sort_values("language")
best.to_csv("baseline/output/qwen2b_best_prompt_by_language.csv", index=False)

# 6) Compact poster table: only prompts most likely to be shown on poster
poster_prompts = ["p0_short_literal", "p0_medium_literal", "p0_long_literal", "p3_object_action", "p4_direct_target"]
poster = df[df["prompt"].isin(poster_prompts)].pivot(index="language", columns="prompt", values="final_chrf")
poster = poster.reindex(columns=poster_prompts)
poster.reset_index().to_csv("baseline/output/qwen2b_compact_poster_table.csv", index=False)

print("Created:")
for f in [
    "baseline/output/qwen2b_poster_complete_summary_sorted.csv",
    "baseline/output/qwen2b_caption_length_effect.csv",
    "baseline/output/qwen2b_prompt_strategy_faceoff.csv",
    "baseline/output/qwen2b_direct_target_failure.csv",
    "baseline/output/qwen2b_best_prompt_by_language.csv",
    "baseline/output/qwen2b_compact_poster_table.csv",
]:
    print("-", f)

print("\nBest prompt by language:")
print(best.to_string(index=False))
PY

echo ""
echo "============================================================"
echo "Final summary"
echo "============================================================"
cat baseline/output/qwen2b_poster_complete_summary.csv

echo ""
echo "Caption length effect:"
cat baseline/output/qwen2b_caption_length_effect.csv

echo ""
echo "Prompt strategy face-off:"
cat baseline/output/qwen2b_prompt_strategy_faceoff.csv

echo ""
echo "Direct target failure:"
cat baseline/output/qwen2b_direct_target_failure.csv

echo ""
echo "Best prompt by language:"
cat baseline/output/qwen2b_best_prompt_by_language.csv

echo ""
echo "Compact poster table:"
cat baseline/output/qwen2b_compact_poster_table.csv

echo ""
echo "============================================================"
echo "Creating archive"
echo "============================================================"

tar -czf lrz_dev_qwen2b_full_poster_results.tar.gz \
  baseline/output/lrz_dev_*_2b_*_50* \
  baseline/output/qwen2b_poster_complete_summary.csv \
  baseline/output/qwen2b_poster_complete_summary_sorted.csv \
  baseline/output/qwen2b_caption_length_effect.csv \
  baseline/output/qwen2b_prompt_strategy_faceoff.csv \
  baseline/output/qwen2b_direct_target_failure.csv \
  baseline/output/qwen2b_best_prompt_by_language.csv \
  baseline/output/qwen2b_compact_poster_table.csv

echo ""
echo "Archive created:"
ls -lh lrz_dev_qwen2b_full_poster_results.tar.gz

echo ""
echo "All done. Download this file:"
echo "lrz_dev_qwen2b_full_poster_results.tar.gz"
echo ""
echo "Most important CSVs for poster:"
echo "baseline/output/qwen2b_compact_poster_table.csv"
echo "baseline/output/qwen2b_caption_length_effect.csv"
echo "baseline/output/qwen2b_prompt_strategy_faceoff.csv"
echo "baseline/output/qwen2b_direct_target_failure.csv"
echo "baseline/output/qwen2b_best_prompt_by_language.csv"
