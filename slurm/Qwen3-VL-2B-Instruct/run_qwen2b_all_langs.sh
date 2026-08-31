#!/bin/bash
set -e

cd ~/americasnlp2026

DSS="<LRZ_DSS_PATH>"
export HF_HOME="$DSS/<LRZ_ACCOUNT>/hf_cache"
export HF_HUB_CACHE="$DSS/<LRZ_ACCOUNT>/hf_cache/hub"
export TRANSFORMERS_CACHE="$DSS/<LRZ_ACCOUNT>/hf_cache"

MODEL="Qwen/Qwen3-VL-2B-Instruct"
PROMPT="p0_long_literal"
SAMPLES=50

mkdir -p baseline/output

# Fix Guarani path issue if needed
mkdir -p data/dev/guarani/data/guarani
rm -rf data/dev/guarani/data/guarani/images
ln -s ../../images data/dev/guarani/data/guarani/images || true

cat > baseline/output/qwen2b_all_languages_summary.csv <<'CSV'
model,language,split,samples,prompt,workflow,target_code,final_chrf,status
CSV

run_lang () {
  LANG=$1
  TGT=$2

  echo "========================================"
  echo "Running $LANG -> $TGT"
  echo "========================================"

  deactivate 2>/dev/null || true

  python baseline/captioning/run_qwen_prompt_experiment.py \
    --model "$MODEL" \
    --language "$LANG" \
    --split dev \
    --prompt-mode "$PROMPT" \
    --max-samples "$SAMPLES" \
    --max-new-tokens 96 \
    --output-prefix "baseline/output/lrz_dev_${LANG}_2b_${PROMPT}_50"

  wc -l "baseline/output/lrz_dev_${LANG}_2b_${PROMPT}_50.txt"

  source mt_env/bin/activate

  CUDA_VISIBLE_DEVICES="" python baseline/americasnlp-2023-sheffield/translate.py \
    --checkpoint submission_3.pt \
    --input "baseline/output/lrz_dev_${LANG}_2b_${PROMPT}_50.txt" \
    --output "baseline/output/lrz_dev_${LANG}_2b_${PROMPT}_translated_${LANG}_50.txt" \
    --src spa_Latn \
    --tgt "$TGT"

  python baseline/eval.py \
    --dataframe "baseline/output/lrz_dev_${LANG}_2b_${PROMPT}_50.jsonl" \
    --translations "baseline/output/lrz_dev_${LANG}_2b_${PROMPT}_translated_${LANG}_50.txt" \
    | tee "baseline/output/lrz_dev_${LANG}_2b_${PROMPT}_final_score.txt"

  SCORE=$(grep "Mean chrF++ score:" "baseline/output/lrz_dev_${LANG}_2b_${PROMPT}_final_score.txt" | awk -F': ' '{print $2}')

  echo "$MODEL,$LANG,dev,$SAMPLES,$PROMPT,Spanish->Sheffield MT,$TGT,$SCORE,completed" >> baseline/output/qwen2b_all_languages_summary.csv

  deactivate 2>/dev/null || true
}

run_lang wixarika hch_Latn
run_lang bribri bzd_Latn
run_lang guarani grn_Latn
run_lang nahuatl nah_Latn

echo "========================================"
echo "ALL DONE"
echo "========================================"
cat baseline/output/qwen2b_all_languages_summary.csv
