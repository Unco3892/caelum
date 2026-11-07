#!/bin/bash --login
# =============================================================================
# CAELUM — Gemini 2.5 Pro clarifier experiment
#
# Tests whether a stronger API clarifier (Gemini 2.5 Pro for Stage 1) improves
# the aleatoric/epistemic decomposition vs. self-clarification.
# Answers are always sampled from the local model (Stage 2 unchanged).
#
# 3 datasets × 3 models = 9 tasks, single seed 42.
# Cache files are tagged *_gemini_pro_* to avoid clobbering Flash Lite runs.
#
# Task mapping (task_id = model_idx * 3 + dataset_idx):
#   model 0: Qwen3-8B
#   model 1: Ministral-3-8B-Instruct-2512
#   model 2: Gemma-4-E4B-it
#   dataset 0: triviaqa   n=300
#   dataset 1: ambigqa    n=200
#   dataset 2: popqa      n=500
#
# Initial submit (all 9 tasks):
#   sbatch --array=0-8 slurm/gemini_pro_clarif.sh
#
# Resume / rerun a single task:
#   sbatch --array=2   slurm/gemini_pro_clarif.sh
# =============================================================================
#SBATCH --job-name caelum-gemini-pro
#SBATCH --partition gpu-h100,gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --cpus-per-task 12
#SBATCH --mem 64G
#SBATCH --time 04:00:00
#SBATCH --output /scratch/%u/caelum/logs/%A_%a_gemini_pro.out
#SBATCH --error  /scratch/%u/caelum/logs/%A_%a_gemini_pro.err

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT to the project directory on the cluster (see slurm/README.md)}"
ENV_PREFIX="${ENV_PREFIX:-$WORK_ROOT/envs/caelum}"
SCRATCH="${SCRATCH:-/scratch/$USER}"
CAELUM_DIR="$SCRATCH/caelum"
RESULTS_DIR="$CAELUM_DIR/results"
# results/ moved out of src/ on 2026-11-05. Stop instead of silently missing the
# cached generations (and regenerating them) in a checkout that still has src/results.
if [ -d "$CAELUM_DIR/src/results" ]; then
    echo "ERROR: $CAELUM_DIR/src/results still exists. Move its contents to $CAELUM_DIR/results" >&2
    echo "       (if results/ does not exist yet: mv \"$CAELUM_DIR/src/results\" \"$CAELUM_DIR/results\")." >&2
    exit 1
fi
# Cached generations moved to results/generations/ on 2026-11-06. Stop instead of
# missing them (and regenerating them) in a checkout that still has the flat layout.
if compgen -G "$CAELUM_DIR/results/unc_*.json" > /dev/null; then
    echo "ERROR: cached generations are still at the top of $CAELUM_DIR/results. Move them with:" >&2
    echo "       mkdir -p \"$CAELUM_DIR/results/generations\" && mv \"$CAELUM_DIR\"/results/{gen,unc,pipeline}_* \"$CAELUM_DIR/results/generations/\"" >&2
    exit 1
fi
LOG_DIR="$CAELUM_DIR/logs"

# ---------------------------------------------------------------------------
# Decode task ID → (model, dataset, n_samples)
# ---------------------------------------------------------------------------
MODELS=(
    "Qwen/Qwen3-8B"
    "mistralai/Ministral-3-8B-Instruct-2512"
    "google/gemma-4-E4B-it"
)
DATASETS=(triviaqa ambigqa popqa)
N_PER_DS=(  300      200     500)

TASK_ID=${SLURM_ARRAY_TASK_ID}
N_DATASETS=3

MODEL_IDX=$(( TASK_ID / N_DATASETS ))
DS_IDX=$(( TASK_ID % N_DATASETS ))

MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DS_IDX]}
N_SAMPLES=${N_PER_DS[$DS_IDX]}
SEED=42

MODEL_TAG=$(echo "$MODEL" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]' | tr '-' '_')

echo "========================================"
echo "CAELUM — Gemini 2.5 Pro clarifier"
echo "Task:    $TASK_ID  (model_idx=$MODEL_IDX ds_idx=$DS_IDX)"
echo "Model:   $MODEL  ($MODEL_TAG)"
echo "Dataset: $DATASET  n=$N_SAMPLES  seed=$SEED"
echo "Node:    $(hostname)"
echo "========================================"

# ---------------------------------------------------------------------------
# Pre-flight: skip if Gemini Pro gen cache already exists
# Cache filename pattern produced by run_pipeline.py with --clarif_model gemini-2.5-pro:
#   gen_{dataset}_{model_tag}_gemini_pro_{n_samples}.json
# ---------------------------------------------------------------------------
GEN_CACHE="$RESULTS_DIR/generations/gen_${DATASET}_${MODEL_TAG}_gemini_pro_${N_SAMPLES}.json"
if [ -f "$GEN_CACHE" ]; then
    echo "[pre-flight] Gemini Pro gen cache already exists: $GEN_CACHE"
    echo "[pre-flight] Skipping regeneration (use --regenerate flag to override)."
    exit 0
fi

# ---------------------------------------------------------------------------
# Environment — load CUDA before conda
# ---------------------------------------------------------------------------
for p in /etc/profile.d/lmodstacks.sh /etc/profile.d/dcsr_path.sh; do [ -f "$p" ] && source "$p"; done
module purge
module load cuda/12.6.2 || module load cuda/12.4 || true
dcsrsoft use 20241118
module load miniforge3/24.11.3-2
conda_init
conda activate "$ENV_PREFIX"

# ---------------------------------------------------------------------------
# Load credentials from .env (HF_TOKEN, GEMINI_API_KEY, GROQ_API_KEY)
# ---------------------------------------------------------------------------
if [ ! -f "$WORK_ROOT/.env" ]; then
    echo "ERROR: $WORK_ROOT/.env not found — GEMINI_API_KEY unavailable. Aborting."
    exit 1
fi
set -a; source "$WORK_ROOT/.env"; set +a

# Fail fast if GEMINI_API_KEY is missing — avoids wasting GPU time
if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "ERROR: GEMINI_API_KEY is not set in $WORK_ROOT/.env. Aborting."
    exit 1
fi

[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN

mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'CUDA avail: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0)}')"

# Verify openai SDK is present (APIClarifier uses the OpenAI-compatible endpoint)
python -c "import openai; print(f'openai SDK version: {openai.__version__}')" || {
    echo "ERROR: openai SDK not found in conda env. Install with: pip install openai"
    exit 1
}

# ---------------------------------------------------------------------------
# HuggingFace cache → scratch
# ---------------------------------------------------------------------------
export HF_HOME="$SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$SCRATCH/huggingface"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1

# ---------------------------------------------------------------------------
# Dataset-specific path overrides
# ---------------------------------------------------------------------------
EXTRA_ARGS=""
case "$DATASET" in
    triviaqa)
        EXTRA_ARGS="--triviaqa_path $CAELUM_DIR/data/eval/TriviaQA.json"
        ;;
    ambigqa)
        EXTRA_ARGS="--ambigqa_path $CAELUM_DIR/data/eval/AmbigQA.json"
        ;;
    popqa)
        EXTRA_ARGS="--popqa_path $CAELUM_DIR/data/eval/PopQA.json"
        ;;
esac

# ---------------------------------------------------------------------------
# Run pipeline — Gemini 2.5 Pro handles Stage 1 clarifications only
# ---------------------------------------------------------------------------
cd "$CAELUM_DIR"

python src/run_pipeline.py \
    --dataset          "$DATASET" \
    --model_name       "$MODEL" \
    --n_samples        "$N_SAMPLES" \
    --seed             "$SEED" \
    --uncertainty_method vne \
    --weight_fn        additive \
    --backend          hf \
    --clarif_provider  gemini \
    --clarif_model     gemini-2.5-pro \
    $EXTRA_ARGS

EXIT_CODE=$?

# ---------------------------------------------------------------------------
# Sync results → project space
# ---------------------------------------------------------------------------
PROJ_RESULTS="$WORK_ROOT/caelum/results"
mkdir -p "$PROJ_RESULTS"
rsync -a "$RESULTS_DIR/" "$PROJ_RESULTS/"
echo "[gemini-pro] Results synced to $PROJ_RESULTS  (exit=$EXIT_CODE)"

exit $EXIT_CODE
