#!/bin/bash --login
# =============================================================================
# CAELUM — gpt-oss-120b generation cell via vLLM (single dataset per job)
#
# Adds a FOURTH model family (OpenAI) at 117B total / 5.1B active params.
# MXFP4 weights (~63GB) fit a single 96GB H100, so jobs schedule instantly,
# and vLLM's continuous batching makes the 50-samples-per-question pattern
# fast. Uses the separate vLLM environment ($VLLM_ENV_PREFIX).
#
# Usage (per dataset, set via --export):
#   sbatch --time=04:00:00 --job-name=caelum-oss120b-triviaqa \
#     --export=ALL,DATASET=triviaqa,N_SAMPLES=300 slurm/gpt_oss_120b_cell.sh
# =============================================================================
#SBATCH --job-name caelum-oss120b
#SBATCH --partition gpu-h100
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --cpus-per-task 16
#SBATCH --mem 128G
#SBATCH --time 04:00:00
#SBATCH --output /scratch/%u/caelum/logs/%A-%x.out
#SBATCH --error  /scratch/%u/caelum/logs/%A-%x.err

WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT to the project directory on the cluster (see slurm/README.md)}"
ENV_PREFIX="${VLLM_ENV_PREFIX:-$WORK_ROOT/envs/caelum_vllm}"
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

MODEL="openai/gpt-oss-120b"
: "${DATASET:?must set DATASET via --export}"
: "${N_SAMPLES:?must set N_SAMPLES via --export}"

echo "========================================"
echo "CAELUM gpt-oss-120b cell (vLLM)"
echo "Dataset: $DATASET  n=$N_SAMPLES"
echo "Node:    $(hostname)   Job: $SLURM_JOB_ID"
echo "========================================"

for p in /etc/profile.d/lmodstacks.sh /etc/profile.d/dcsr_path.sh; do [ -f "$p" ] && source "$p"; done
module purge
dcsrsoft use 20260114
# The 20260114 stack does not put nvcc on PATH by itself; without this,
# vLLM dies at first inference with "Could not find nvcc".
module load cuda/13.0.2
module load miniforge3/25.3.0-3
conda_init
conda activate "$ENV_PREFIX"

if [ -f "$WORK_ROOT/.env" ]; then
    set -a; source "$WORK_ROOT/.env"; set +a
fi
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export HF_HOME="$SCRATCH/huggingface"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1

# Single-GPU vLLM; leave headroom for the MXFP4 dequant workspace. Reasoning
# models spend tokens on the analysis channel before the final answer, so
# give the sampler enough room (the wrapper strips the scaffolding).
export CAELUM_VLLM_TP=1
export CAELUM_VLLM_PP=1
export CAELUM_VLLM_GPU_UTIL=0.90
export CAELUM_VLLM_MAX_LEN=4096

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
    --backend             vllm \
    $EXTRA_ARGS

EXIT_CODE=$?

PROJ_RESULTS="$WORK_ROOT/caelum/results"
mkdir -p "$PROJ_RESULTS"
rsync -a "$RESULTS_DIR/" "$PROJ_RESULTS/"
echo "[gpt-oss-120b] Results mirrored to $PROJ_RESULTS  (exit=$EXIT_CODE)"
exit $EXIT_CODE
