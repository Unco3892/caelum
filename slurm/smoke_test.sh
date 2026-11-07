#!/bin/bash --login
# =============================================================================
# CAELUM — Stage 1 smoke test
# 1 model × 1 dataset (ambiginst, 50 samples) × 1 seed
# Validates: SLURM, conda, model load, generation, output paths
#
# Submit: sbatch slurm/smoke_test.sh
# =============================================================================
#SBATCH --job-name caelum-smoke
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --cpus-per-task 12
#SBATCH --mem 32G
#SBATCH --time 04:00:00
#SBATCH --output /scratch/%u/caelum/logs/smoke_%j.out
#SBATCH --error  /scratch/%u/caelum/logs/smoke_%j.err

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

MODEL="Qwen/Qwen3-8B"
DATASET="ambiginst"
N_SAMPLES=50
SEED=42

# AmbigInst lives in the spectral data tree
AMBIGINST_PATH="$CAELUM_DIR/data/eval/AmbigInst.json"

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

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
echo "========================================"
echo "CAELUM smoke test"
echo "Job:     $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Model:   $MODEL"
echo "Dataset: $DATASET  n=$N_SAMPLES  seed=$SEED"
echo "========================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}, Device: {torch.cuda.get_device_name(0)}')"

# ---------------------------------------------------------------------------
# HuggingFace cache → scratch (fast SSD, avoids project quota pressure)
# ---------------------------------------------------------------------------
export HF_HOME="$SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$SCRATCH/huggingface"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1

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
    --ambiginst_path "$AMBIGINST_PATH"

EXIT_CODE=$?

# ---------------------------------------------------------------------------
# Sync results back to project space
# ---------------------------------------------------------------------------
PROJ_RESULTS="$WORK_ROOT/caelum/results"
mkdir -p "$PROJ_RESULTS"
rsync -a "$RESULTS_DIR/" "$PROJ_RESULTS/"
echo "[smoke] Results synced to $PROJ_RESULTS"

echo "[smoke] Finished with exit code $EXIT_CODE"
exit $EXIT_CODE
