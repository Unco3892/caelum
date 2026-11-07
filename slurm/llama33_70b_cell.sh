#!/bin/bash --login
# =============================================================================
# CAELUM — Llama-3.3-70B generation cell (single dataset per submission)
#
# Extends the scale audit to 70B AND adds a fourth model family (Meta):
# the 8B->32B judged matchability delta is +2.8..+3.6pp on the
# knowledge-bound cells, and this row tests whether the support ceiling
# stays knowledge-bound at 70B in a different pretraining family.
# bf16 weights are ~140GB, so this shards across 2x80GB H100 via
# device_map=auto.
#
# Usage (per dataset, set via --export):
#   sbatch --time=06:00:00 --job-name=caelum-l70b-triviaqa \
#     --export=ALL,DATASET=triviaqa,N_SAMPLES=300 slurm/llama33_70b_cell.sh
#   (use --time=10:00:00 for the 500-sample datasets)
# =============================================================================
#SBATCH --job-name caelum-l70b
#SBATCH --partition gpu-h100
#SBATCH --gres gpu:2
#SBATCH --gres-flags enforce-binding
#SBATCH --cpus-per-task 24
#SBATCH --mem 192G
#SBATCH --time 06:00:00
#SBATCH --output /scratch/%u/caelum/logs/%A-%x.out
#SBATCH --error  /scratch/%u/caelum/logs/%A-%x.err

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

MODEL="meta-llama/Llama-3.3-70B-Instruct"
: "${DATASET:?must set DATASET via --export}"
: "${N_SAMPLES:?must set N_SAMPLES via --export}"

echo "========================================"
echo "CAELUM Llama-3.3-70B cell"
echo "Dataset: $DATASET  n=$N_SAMPLES"
echo "Node:    $(hostname)"
echo "Job:     $SLURM_JOB_ID"
echo "========================================"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
for p in /etc/profile.d/lmodstacks.sh /etc/profile.d/dcsr_path.sh; do [ -f "$p" ] && source "$p"; done
module purge
module load cuda/12.6.2 || module load cuda/12.4 || true
dcsrsoft use 20241118
module load miniforge3/24.11.3-2
conda_init
conda activate "$ENV_PREFIX"

if [ -f "$WORK_ROOT/.env" ]; then
    set -a; source "$WORK_ROOT/.env"; set +a
fi
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'CUDA avail: {torch.cuda.is_available()}, n_gpu={torch.cuda.device_count()}')"

export HF_HOME="$SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$SCRATCH/huggingface"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1

# bf16 sharded across both H100s (~70GB per card + activations); the
# built-in OOM retry halves the batch if activations push past the limit.
export CAELUM_HF_USE_4BIT=0
export CAELUM_HF_DEVICE_MAP_AUTO=1

# ---------------------------------------------------------------------------
# Dataset-specific path overrides (mirrors slurm/full_benchmark.sh convention)
# ---------------------------------------------------------------------------
EXTRA_ARGS=""
case "$DATASET" in
    ambiginst) EXTRA_ARGS="--ambiginst_path $CAELUM_DIR/data/eval/AmbigInst.json" ;;
    triviaqa)  EXTRA_ARGS="--triviaqa_path $CAELUM_DIR/data/eval/TriviaQA.json" ;;
    ambigqa)   EXTRA_ARGS="--ambigqa_path $CAELUM_DIR/data/eval/AmbigQA.json" ;;
    opennq)    EXTRA_ARGS="--opennq_path $CAELUM_DIR/data/eval/OpenNQ.json" ;;
    simpleqa)  EXTRA_ARGS="--simpleqa_path $CAELUM_DIR/data/eval/SimpleQA.json" ;;
    popqa)     EXTRA_ARGS="--popqa_path $CAELUM_DIR/data/eval/PopQA.json" ;;
esac

cd "$CAELUM_DIR"

python src/run_pipeline.py \
    --dataset            "$DATASET" \
    --model_name         "$MODEL" \
    --n_samples           "$N_SAMPLES" \
    --uncertainty_method  vne \
    --n_clarifications    5 \
    --n_answers           10 \
    --cluster_threshold   0.7 \
    --weight_fn           additive \
    --backend             hf \
    $EXTRA_ARGS

EXIT_CODE=$?

PROJ_RESULTS="$WORK_ROOT/caelum/results"
mkdir -p "$PROJ_RESULTS"
rsync -a "$RESULTS_DIR/" "$PROJ_RESULTS/"
echo "[llama33-70b] Results mirrored to $PROJ_RESULTS  (exit=$EXIT_CODE)"

exit $EXIT_CODE
