#!/bin/bash --login
# =============================================================================
# CAELUM — Full benchmark array job
#
# Array indexing (task_id = seed_idx * 18 + model_idx * 6 + dataset_idx):
#   seed 0 (42): tasks  0-17
#   seed 1 (43): tasks 18-35
#   ...
#   seed 4 (46): tasks 72-89
#
# Initial run (1 seed, 18 tasks):
#   sbatch --array=0-17  slurm/full_benchmark.sh
#
# Full 5-seed run (90 tasks):
#   sbatch --array=0-89  slurm/full_benchmark.sh
#
# Resume after partial completion (skip cached gens automatically — see pre-flight):
#   sbatch --array=0-17  slurm/full_benchmark.sh
# =============================================================================
#SBATCH --job-name caelum-bench
# Use H100 partition first, A100 (gpu) as fallback. SLURM picks whichever
# can satisfy the request first, maximizing throughput.
#SBATCH --partition gpu-h100,gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --cpus-per-task 12
#SBATCH --mem 64G
#SBATCH --time 04:00:00
#SBATCH --output /scratch/%u/caelum/logs/%A_%a.out
#SBATCH --error  /scratch/%u/caelum/logs/%A_%a.err

# ---------------------------------------------------------------------------
# Config — WORK_ROOT (required) and ENV_PREFIX/SCRATCH (optional) come from the environment
# ---------------------------------------------------------------------------
WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT to the project directory on the cluster (see slurm/README.md)}"
ENV_PREFIX="${ENV_PREFIX:-$WORK_ROOT/envs/caelum}"
# SCRATCH is not auto-exported in DCSR SLURM jobs; set it explicitly.
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
# Decode task ID
# ---------------------------------------------------------------------------
# Note: CondAmbigQA dropped from the benchmark — questions trigger 2-4x longer
# generations than other datasets, causing 4h+ per (model, dataset) cell with
# Ministral consistently timing out. AmbigQA + AmbigInst already cover the
# ambiguous regime; CondAmbigQA was redundant.
DATASETS=(triviaqa ambigqa opennq ambiginst simpleqa popqa)
# Subset sizes match the curated spectral evaluation set (where applicable):
# - TriviaQA, OpenNQ:  300 items (spectral subset)
# - AmbigQA:           200 items (spectral subset)
# - AmbigInst:         304 items (full spectral set)
# - SimpleQA, PopQA: 500 items (subsampled from raw, no spectral curation)
N_PER_DS=(  300      200     300     304      500     500)
MODELS=(
    "Qwen/Qwen3-8B"
    "mistralai/Ministral-3-8B-Instruct-2512"
    "google/gemma-4-E4B-it"
)
SEEDS=(42 43 44 45 46)

TASK_ID=${SLURM_ARRAY_TASK_ID}
N_DATASETS=6
N_MODELS=3

SEED_IDX=$(( TASK_ID / (N_DATASETS * N_MODELS) ))
REMAINDER=$(( TASK_ID % (N_DATASETS * N_MODELS) ))
MODEL_IDX=$(( REMAINDER / N_DATASETS ))
DS_IDX=$(( REMAINDER % N_DATASETS ))

SEED=${SEEDS[$SEED_IDX]}
MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DS_IDX]}
N_SAMPLES=${N_PER_DS[$DS_IDX]}

MODEL_TAG=$(echo "$MODEL" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]' | tr '-' '_')

echo "========================================"
echo "CAELUM benchmark"
echo "Task:    $TASK_ID  (seed_idx=$SEED_IDX model_idx=$MODEL_IDX ds_idx=$DS_IDX)"
echo "Model:   $MODEL  ($MODEL_TAG)"
echo "Dataset: $DATASET  n=$N_SAMPLES  seed=$SEED"
echo "Node:    $(hostname)"
echo "========================================"

# ---------------------------------------------------------------------------
# Pre-flight: skip if gen cache already exists for this exact combination
# (Cache filename format matches run_pipeline.py logic:
#  gen_{dataset}_{model_tag}_{n_samples}.json — seed not in gen cache name,
#  so each (model, dataset) generates once and seeds reuse it for calibration.)
# ---------------------------------------------------------------------------
GEN_CACHE="$RESULTS_DIR/generations/gen_${DATASET}_${MODEL_TAG}_${N_SAMPLES}.json"

# The unc cache DOES include seed (via data reshuffling), so check unc file
# using a seed-tagged name if you add --seed to the unc path. For now we
# treat the seed as only affecting calibration/split; generation is reused.
# If gen cache exists AND we are on a non-zero seed, generation is skipped
# automatically by run_pipeline.py (cache hit). Still run to get unc+results.
if [ -f "$GEN_CACHE" ] && [ "$SEED_IDX" -eq 0 ]; then
    echo "[pre-flight] Gen cache exists and this is seed 0 — nothing to do, exiting."
    exit 0
fi

# ---------------------------------------------------------------------------
# Environment — load CUDA before conda
# ---------------------------------------------------------------------------
# Source DCSR shell functions (dcsrsoft is a bash function)
for p in /etc/profile.d/lmodstacks.sh /etc/profile.d/dcsr_path.sh; do [ -f "$p" ] && source "$p"; done
module purge
module load cuda/12.6.2 || module load cuda/12.4 || true
dcsrsoft use 20241118
module load miniforge3/24.11.3-2
conda_init
conda activate "$ENV_PREFIX"

# Load HF token for gated models
if [ -f "$WORK_ROOT/.env" ]; then
    set -a; source "$WORK_ROOT/.env"; set +a
fi
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN

mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'CUDA avail: {torch.cuda.is_available()}')"

# ---------------------------------------------------------------------------
# HF cache on scratch
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
    ambiginst)
        EXTRA_ARGS="--ambiginst_path $CAELUM_DIR/data/eval/AmbigInst.json"
        ;;
    triviaqa)
        EXTRA_ARGS="--triviaqa_path $CAELUM_DIR/data/eval/TriviaQA.json"
        ;;
    ambigqa)
        EXTRA_ARGS="--ambigqa_path $CAELUM_DIR/data/eval/AmbigQA.json"
        ;;
    opennq)
        EXTRA_ARGS="--opennq_path $CAELUM_DIR/data/eval/OpenNQ.json"
        ;;
    simpleqa)
        EXTRA_ARGS="--simpleqa_path $CAELUM_DIR/data/eval/SimpleQA.json"
        ;;
    popqa)
        EXTRA_ARGS="--popqa_path $CAELUM_DIR/data/eval/PopQA.json"
        ;;
    condambigqa)
        EXTRA_ARGS="--condambigqa_path $CAELUM_DIR/data/eval/CondAmbigQA.json"
        ;;
esac

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
cd "$CAELUM_DIR"

python src/run_pipeline.py \
    --dataset        "$DATASET" \
    --model_name     "$MODEL" \
    --n_samples      "$N_SAMPLES" \
    --seed           "$SEED" \
    --uncertainty_method vne \
    --weight_fn      additive \
    --backend        hf \
    $EXTRA_ARGS

EXIT_CODE=$?

# ---------------------------------------------------------------------------
# Sync results → project space
# ---------------------------------------------------------------------------
PROJ_RESULTS="$WORK_ROOT/caelum/results"
mkdir -p "$PROJ_RESULTS"
rsync -a "$RESULTS_DIR/" "$PROJ_RESULTS/"
echo "[bench] Results synced to $PROJ_RESULTS  (exit=$EXIT_CODE)"

exit $EXIT_CODE
