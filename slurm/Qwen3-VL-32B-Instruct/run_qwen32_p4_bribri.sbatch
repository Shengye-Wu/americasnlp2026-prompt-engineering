#!/bin/bash
#SBATCH --job-name=qwen32_eval
#SBATCH -p lrz-hgx-a100-80x4
#SBATCH --output=logs/qwen32_eval_%j.out
#SBATCH --error=logs/qwen32_eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --gres=gpu:2

cd ~/americasnlp2026

DSS="/dss/dssfs05/lwp-dss-0003/<LRZ_PROJECT_ID>/<LRZ_PROJECT_DSS_ID>"
MODEL32="Qwen/Qwen3-VL-32B-Instruct"

export HF_HOME="$DSS/<LRZ_ACCOUNT>/hf_cache"
export HF_HUB_CACHE="$DSS/<LRZ_ACCOUNT>/hf_cache/hub"
export TRANSFORMERS_CACHE="$DSS/<LRZ_ACCOUNT>/hf_cache"

echo "Job started on $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# ==========================================
# Dynamic Experiment Parameter Configuration Area (Modify here to control the pipeline)
# ==========================================
EXP_LANG="bribri"
EXP_SPLIT="dev"
EXP_PROMPT="p4_direct_target"  # <--- To test a new strategy later, just modify this line
EXP_SAMPLES="50"
EXP_MODEL_ALIAS="32b"
SRC_CODE="spa_Latn"
TGT_CODE="bzd_Latn"                # If changing languages, remember to update this to the corresponding language code

# Automatically concatenate to generate a dynamic output prefix
# Example output: baseline/output/lrz_dev_wixarika_32b_p5_visual_dictionary_50
OUT_PREFIX="baseline/output/lrz_${EXP_SPLIT}_${EXP_LANG}_${EXP_MODEL_ALIAS}_${EXP_PROMPT}_${EXP_SAMPLES}"

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
SECONDS=0

echo "=========================================="
echo "Starting Automated Pipeline | Experiment Prefix: $OUT_PREFIX"
echo "馃 Prompt Strategy : $EXP_PROMPT"
echo "Output Prefix   : $OUT_PREFIX"
echo "Start Time      : $START_TIME"
echo "=========================================="

# 1. Run LLM Image Captioning
python scripts/run_qwen_prompt_experiment.py \
  --model "$MODEL32" \
  --language "$EXP_LANG" \
  --split "$EXP_SPLIT" \
  --prompt-mode "$EXP_PROMPT" \
  --max-samples "$EXP_SAMPLES" \
  --max-new-tokens 96 \
  --output-prefix "$OUT_PREFIX"

echo "Captioning finished"

wc -l ${OUT_PREFIX}.txt
head -n 5 ${OUT_PREFIX}.txt

source mt_env/bin/activate

# 2. Run Machine Translation (Filenames automatically include the Prompt identifier)
CUDA_VISIBLE_DEVICES="" python baseline/americasnlp-2023-sheffield/translate.py \
  --checkpoint submission_3.pt \
  --input ${OUT_PREFIX}.txt \
  --output ${OUT_PREFIX}_translated.txt \
  --src "$SRC_CODE" \
  --tgt "$TGT_CODE"

# 3. Run Final Scoring (JSONL and TXT matched automatically and precisely)
python baseline/eval.py \
  --dataframe ${OUT_PREFIX}.jsonl \
  --translations ${OUT_PREFIX}_translated.txt \
  | tee ${OUT_PREFIX}_final_score.txt

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
DURATION=$SECONDS
HOURS=$(($DURATION / 3600))
MINUTES=$((($DURATION % 3600) / 60))
SECS=$(($DURATION % 60))

echo "=========================================="
echo "Pipeline Finished!"
echo "Final score for $EXP_PROMPT:"
cat ${OUT_PREFIX}_final_score.txt
echo "End Time        : $END_TIME"
echo "Total Duration  : ${HOURS}h ${MINUTES}m ${SECS}s"
echo "=========================================="
