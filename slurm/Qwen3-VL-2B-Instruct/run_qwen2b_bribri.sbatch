#!/bin/bash
#SBATCH --job-name=qwen2b_bribri_eval
#SBATCH -p lrz-dgx-1-v100x8
#SBATCH --output=logs/qwen2b_bribri_%j.out
#SBATCH --error=logs/qwen2b_bribri_%j.err
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:2

cd ~/americasnlp2026

DSS="/dss/dssfs05/lwp-dss-0003/<LRZ_PROJECT_ID>/<LRZ_PROJECT_DSS_ID>"
MODEL="Qwen/Qwen3-VL-2B-Instruct"

export HF_HOME="$DSS/<LRZ_ACCOUNT>/hf_cache"
export HF_HUB_CACHE="$DSS/<LRZ_ACCOUNT>/hf_cache/hub"

mkdir -p logs baseline/output

EXP_LANG="bribri"
EXP_SPLIT="dev"
EXP_SAMPLES="50"
EXP_MODEL_ALIAS="2b"
SRC_CODE="spa_Latn"
TGT_CODE="bzd_Latn"

PROMPT_LIST=(
  "p0_short_literal"
  "p0_medium_literal"
  "p0_long_literal"
  "p1_culture_aware"
  "p2_translation_friendly"
  "p2b_translation_friendly_detailed"
  "p3_object_action"
  "p4_direct_target"
)

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
SECONDS=0

for EXP_PROMPT in "${PROMPT_LIST[@]}"; do
  OUT_PREFIX="baseline/output/lrz_${EXP_SPLIT}_${EXP_LANG}_${EXP_MODEL_ALIAS}_${EXP_PROMPT}_${EXP_SAMPLES}"

  python scripts/run_qwen_prompt_experiment.py \
    --model "$MODEL" \
    --language "$EXP_LANG" \
    --split "$EXP_SPLIT" \
    --prompt-mode "$EXP_PROMPT" \
    --max-samples "$EXP_SAMPLES" \
    --max-new-tokens 96 \
    --output-prefix "$OUT_PREFIX"

  if [ "$EXP_PROMPT" = "p4_direct_target" ]; then
    source mt_env/bin/activate
    python baseline/eval.py \
      --dataframe "${OUT_PREFIX}.jsonl" \
      --translations "${OUT_PREFIX}.txt" \
      | tee "${OUT_PREFIX}_final_score.txt"
    deactivate
  else
    source mt_env/bin/activate
    CUDA_VISIBLE_DEVICES="" python baseline/americasnlp-2023-sheffield/translate.py \
      --checkpoint submission_3.pt \
      --input "${OUT_PREFIX}.txt" \
      --output "${OUT_PREFIX}_translated.txt" \
      --src "$SRC_CODE" \
      --tgt "$TGT_CODE"

    python baseline/eval.py \
      --dataframe "${OUT_PREFIX}.jsonl" \
      --translations "${OUT_PREFIX}_translated.txt" \
      | tee "${OUT_PREFIX}_final_score.txt"
    deactivate
  fi
done

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
DURATION=$SECONDS
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECS=$((DURATION % 60))

echo "Start Time     : $START_TIME"
echo "End Time       : $END_TIME"
echo "Total Duration : ${HOURS}h ${MINUTES}m ${SECS}s"

echo "Summary of Results:"
for EXP_PROMPT in "${PROMPT_LIST[@]}"; do
  OUT_PREFIX="baseline/output/lrz_${EXP_SPLIT}_${EXP_LANG}_${EXP_MODEL_ALIAS}_${EXP_PROMPT}_${EXP_SAMPLES}"
  if [ -f "${OUT_PREFIX}_final_score.txt" ]; then
    echo " - $EXP_PROMPT: $(cat "${OUT_PREFIX}_final_score.txt")"
  fi
done
