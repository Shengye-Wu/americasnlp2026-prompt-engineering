#!/bin/bash
#SBATCH --job-name=qwen2b_pilot_comet
#SBATCH -p lrz-dgx-1-v100x8
#SBATCH --output=logs/qwen2b_eval_%j.out
#SBATCH --error=logs/qwen2b_eval_%j.err
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:2

cd ~/americasnlp2026

DSS="/dss/dssfs05/lwp-dss-0003/<LRZ_PROJECT_ID>/<LRZ_PROJECT_DSS_ID>"
MODEL32="Qwen/Qwen3-VL-2B-Instruct"

export HF_HOME="$DSS/<LRZ_ACCOUNT>/hf_cache"
export HF_HUB_CACHE="$DSS/<LRZ_ACCOUNT>/hf_cache/hub"

echo "Job started on $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# ==========================================
# Static Experiment Parameters
# ==========================================
EXP_LANG="wixarika"
EXP_SPLIT="pilot"    # Changed from dev to pilot
EXP_SAMPLES="20"     # Changed to 20 to match the pilot set size
EXP_MODEL_ALIAS="2b"
SRC_CODE="spa_Latn"
TGT_CODE="hch_Latn"

# The ground-truth Spanish reference file from the pilot set
REF_FILE="data/${EXP_SPLIT}/${EXP_LANG}/${EXP_LANG}_spa.txt"
RESULTS_DIR="comet_results/${EXP_LANG}_pilot"
mkdir -p "$RESULTS_DIR"

# ==========================================
# Loop Configuration
# ==========================================
PROMPT_LIST=(
    "p0_short_literal"
    "p0_medium_literal"
    "p0_long_literal"
    "p1_culture_aware"
    "p2_translation_friendly"
    "p2b_translation_friendly_detailed"
    "p3_object_action"
)

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
SECONDS=0

echo "=========================================="
echo "Starting Automated Pilot Pipeline"
echo "Start Time: $START_TIME"
echo "=========================================="

for EXP_PROMPT in "${PROMPT_LIST[@]}"; do

  OUT_PREFIX="baseline/output/lrz_${EXP_SPLIT}_${EXP_LANG}_${EXP_MODEL_ALIAS}_${EXP_PROMPT}_${EXP_SAMPLES}"
  RESULTS_TXT="${RESULTS_DIR}/lrz_${EXP_SPLIT}_${EXP_LANG}_${EXP_MODEL_ALIAS}_${EXP_PROMPT}_${EXP_SAMPLES}_scores.txt"
  RESULTS_JSON="${RESULTS_DIR}/lrz_${EXP_SPLIT}_${EXP_LANG}_${EXP_MODEL_ALIAS}_${EXP_PROMPT}_${EXP_SAMPLES}_scores.json"

  echo ""
  echo "------------------------------------------------------------"
  echo "Running Prompt Strategy: $EXP_PROMPT"
  echo "Output Prefix: $OUT_PREFIX"
  echo "------------------------------------------------------------"

  # 1. Run LLM Image Captioning (Generates Spanish)
  echo "[1/3] Running Captioning..."
  python baseline/captioning/run_qwen_prompt_experiment.py \
    --model "$MODEL32" \
    --language "$EXP_LANG" \
    --split "$EXP_SPLIT" \
    --prompt-mode "$EXP_PROMPT" \
    --max-samples "$EXP_SAMPLES" \
    --max-new-tokens 96 \
    --output-prefix "$OUT_PREFIX"

  echo "Captioning finished"
  wc -l ${OUT_PREFIX}.txt

  # Activate the MT environment
  source mt_env/bin/activate

  # 3. Run COMET Evaluation on Spanish Pilot Data
  echo "[3/3] Running COMET Evaluation on Spanish references..."
  # We evaluate the generated Spanish (${OUT_PREFIX}.txt) against the reference Spanish ($REF_FILE)
  # Using a dummy source or identical source since this is monolingual comparison
  ~/.local/bin/comet-score \
    -s "$REF_FILE" \
    -t "${OUT_PREFIX}.txt" \
    -r "$REF_FILE" \
    --model Unbabel/wmt22-comet-da \
    --to_json "$RESULTS_JSON" > "$RESULTS_TXT"

  deactivate
  
  echo "Score for $EXP_PROMPT recorded in $RESULTS_TXT!"
  tail -n 1 "$RESULTS_TXT"

done

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
DURATION=$SECONDS

echo ""
echo "=========================================="
echo "Pipeline Loop Finished!"
echo "End Time        : $END_TIME"
echo "=========================================="
