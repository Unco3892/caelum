#!/bin/bash --login
# =============================================================================
# CAELUM — Qwen3-235B-A22B generation cell (single dataset per submission)
#
# Extends the within-family scale audit to 235B total / 22B active
# parameters: judged matchability moved +2.8..+3.6pp on the knowledge-bound
# cells from 8B to 32B, and this row tests whether the support ceiling is
# still knowledge-bound at frontier open-weight scale. bf16 weights are
# ~470GB, so the model loads 4-bit via bitsandbytes (~120GB) sharded across
# 2x80GB H100 with device_map=auto; the quantization caveat is noted in the
# paper's limitations.
#
# Usage (per dataset, set via --export):
#   sbatch --time=08:00:00 --job-name=caelum-q235b-triviaqa \
#     --export=ALL,DATASET=triviaqa,N_SAMPLES=300 slurm/qwen3_235b_cell.sh
#   (use --time=12:00:00 for the 500-sample datasets)
# =============================================================================
#SBATCH --job-name caelum-q235b
#SBATCH --partition gpu-h100
#SBATCH --gres gpu:3
#SBATCH --gres-flags enforce-binding
#SBATCH --cpus-per-task 24
#SBATCH --mem 256G
#SBATCH --time 08:00:00
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

MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
: "${DATASET:?must set DATASET via --export}"
: "${N_SAMPLES:?must set N_SAMPLES via --export}"

echo "========================================"
echo "CAELUM Qwen3-235B-A22B cell"
echo "Dataset: $DATASET  n=$N_SAMPLES"
echo "Node:    $(hostname)"
echo "Job:     $SLURM_JOB_ID"
echo "========================================"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
for p in /etc/profile.d/lmodstacks.sh /etc/profile.d/dcsr_path.sh; do [ -f "$p" ] && source "$p"; done
module purge
# DeepGEMM (the fast FP8 kernel, 3-6x over triton) JIT-compiles at runtime
# and needs an nvcc that matches the kernel build and torch (both cu130):
# nvcc 12.4 from the 20241118 stack fails with "NVCC compilation failed".
# The 20260114 stack ships cuda/13.0.2 — an exact cu130 match (verified
# interactively: nvcc 13.0 + torch 2.11.0+cu130 import cleanly together).
dcsrsoft use 20260114
module load miniforge3/25.3.0-3
module load gcc/11.5.0
module load cuda/13.0.2
echo "CUDA_HOME=${CUDA_HOME:-unset}  nvcc: $(command -v nvcc || echo missing)"
# Keep the env-shipped cudart symlink dir as a fallback for ctypes lookups.
export LD_LIBRARY_PATH="$SCRATCH/caelum/lib:${LD_LIBRARY_PATH:-}"
# DeepGEMM FP8 kernels only emit bf16/fp32; fp16 activations assert-fail.
export CAELUM_HF_DTYPE=bfloat16
# CAELUM_FP8_FORCE_TRITON=1 remains available as a fallback; with the full
# toolkit loaded we let DeepGEMM JIT-compile (measured: triton fallback runs
# ~16 min/question, which is uneconomical).
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

# Official FP8 checkpoint (~240GB), H100-native, auto-detected from the
# checkpoint's quantization_config. bitsandbytes is a dead end here: this
# transformers version fuses MoE experts into 3D nn.Parameter tensors that
# bnb cannot quantize, so the experts stay unquantized (~460GB) under any
# bnb setting. 3x96GB H100 holds the FP8 weights with room for activations.
export CAELUM_HF_USE_4BIT=0
export CAELUM_HF_DEVICE_MAP_AUTO=1
# The partition's H100s are the 96GB NVL variant (95830 MiB). 84GiB per card
# caps weight placement at ~80/card (240GB FP8 over 3 cards) and leaves
# ~15GB/card for activations, KV cache, and triton workspace — 90GiB packed
# the weights so tight that the first generate OOMed inside the kernel.
export CAELUM_HF_MAX_MEMORY_GIB=84

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
echo "[qwen3-235b] Results mirrored to $PROJ_RESULTS  (exit=$EXIT_CODE)"

exit $EXIT_CODE
