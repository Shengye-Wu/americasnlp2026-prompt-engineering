#!/bin/bash
#SBATCH --job-name=qwen8b_eval_loop
#SBATCH -p lrz-dgx-1-v100x8
#SBATCH --output=logs/qwen8b_eval_%j.out
#SBATCH --error=logs/qwen8b_eval_%j.err
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:2

cd ~/americasnlp2026

DSS="/dss/dssfs05/lwp-dss-0003/<LRZ_PROJECT_ID>/<LRZ_PROJECT_DSS_ID>"
MODEL="Qwen/Qwen3-VL-8B-Instruct"

export HF_HOME="$DSS/<LRZ_ACCOUNT>/hf_cache"
export HF_HUB_CACHE="$DSS/<LRZ_ACCOUNT>/hf_cache/hub"
# export TRANSFORMERS_CACHE is removed to avoid the deprecation warning

echo "Job started on $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# ==========================================
# Static Experiment Parameters
# ==========================================
EXP_LANG="wixarika"
EXP_SPLIT="dev"
EXP_SAMPLES="50"
EXP_MODEL_ALIAS="8b"
SRC_CODE="spa_Latn"
TGT_CODE="hch_Latn"  # Updated to Wix谩rika (was bzd_Latn)

# ==========================================
# Loop Configuration (Add your 10 strategies here)
# ==========================================
PROMPT_LIST=(
    "p3_object_action"
    "p4_direct_target"
)

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
SECONDS=0

echo "=========================================="
echo "Starting Automated Pipeline Loop"
echo "Start Time: $START_TIME"
echo "=========================================="

for EXP_PROMPT in "${PROMPT_LIST[@]}"; do

  # Dynamically generate the output prefix for the current prompt
  OUT_PREFIX="baseline/output/lrz_${EXP_SPLIT}_${EXP_LANG}_${EXP_MODEL_ALIAS}_${EXP_PROMPT}_${EXP_SAMPLES}"

  echo ""
  echo "------------------------------------------------------------"
  echo "Running Prompt Strategy: $EXP_PROMPT"
  echo "Output Prefix: $OUT_PREFIX"
  echo "------------------------------------------------------------"

  # 1. Run LLM Image Captioning
  echo "[1/3] Running Captioning..."
  python scripts/run_qwen_prompt_experiment.py \
    --model "$MODEL" \
    --language "$EXP_LANG" \
    --split "$EXP_SPLIT" \
    --prompt-mode "$EXP_PROMPT" \
    --max-samples "$EXP_SAMPLES" \
    --max-new-tokens 96 \
    --output-prefix "$OUT_PREFIX"

  echo "Captioning finished"
  wc -l ${OUT_PREFIX}.txt
  head -n 5 ${OUT_PREFIX}.txt

  # Activate the MT environment specifically for steps 2 and 3
  source mt_env/bin/activate

  # 2. Run Machine Translation
  echo "[2/3] Running Translation..."
  CUDA_VISIBLE_DEVICES="" python baseline/americasnlp-2023-sheffield/translate.py \
    --checkpoint submission_3.pt \
    --input ${OUT_PREFIX}.txt \
    --output ${OUT_PREFIX}_translated.txt \
    --src "$SRC_CODE" \
    --tgt "$TGT_CODE"

  # 3. Run Final Scoring
  echo "[3/3] Running Evaluation..."
  python baseline/eval.py \
    --dataframe ${OUT_PREFIX}.jsonl \
    --translations ${OUT_PREFIX}_translated.txt \
    | tee ${OUT_PREFIX}_final_score.txt

  # Deactivate the MT environment so the next iteration's Step 1 runs in the base environment
  deactivate
  
  echo "Score for $EXP_PROMPT recorded!"

done

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
DURATION=$SECONDS
HOURS=$(($DURATION / 3600))
MINUTES=$((($DURATION % 3600) / 60))
SECS=$(($DURATION % 60))

echo ""
echo "=========================================="
echo "Pipeline Loop Finished for all ${#PROMPT_LIST[@]} prompts!"
echo "End Time        : $END_TIME"
echo "Total Duration  : ${HOURS}h ${MINUTES}m ${SECS}s"
echo "=========================================="

# Print a summary of all scores at the very end
echo "Summary of Results:"
for EXP_PROMPT in "${PROMPT_LIST[@]}"; do
  OUT_PREFIX="baseline/output/lrz_${EXP_SPLIT}_${EXP_LANG}_${EXP_MODEL_ALIAS}_${EXP_PROMPT}_${EXP_SAMPLES}"
  SCORE=$(cat ${OUT_PREFIX}_final_score.txt)
  echo " - $EXP_PROMPT: $SCORE"
done
