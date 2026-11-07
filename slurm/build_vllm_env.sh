#!/bin/bash --login
# =============================================================================
# Build a dedicated vLLM conda env for CAELUM large-model rows.
#
# Kept separate from the main CAELUM environment, because vLLM pins its own
# torch. Target models:
#   - openai/gpt-oss-120b        (MXFP4, 1x96GB H100)
#   - Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 (PP=3 across 3x96GB H100)
#
# Submit:  sbatch slurm/build_vllm_env.sh   (target: $VLLM_ENV_PREFIX)
# =============================================================================
#SBATCH --job-name build-vllm-env
#SBATCH --partition cpu
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 04:00:00
#SBATCH --output /scratch/%u/caelum/logs/build_vllm_env_%j.out
#SBATCH --error  /scratch/%u/caelum/logs/build_vllm_env_%j.err

WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT to the project directory on the cluster (see slurm/README.md)}"
ENV_PREFIX="${VLLM_ENV_PREFIX:-$WORK_ROOT/envs/caelum_vllm}"

for p in /etc/profile.d/lmodstacks.sh /etc/profile.d/dcsr_path.sh; do [ -f "$p" ] && source "$p"; done
module purge
dcsrsoft use 20260114
module load miniforge3/25.3.0-3
conda_init

export TMPDIR="/scratch/$USER/tmp"
export PIP_CACHE_DIR="/scratch/$USER/pip_cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

if [ ! -d "$ENV_PREFIX" ]; then
    conda create -y -p "$ENV_PREFIX" python=3.12
fi
conda activate "$ENV_PREFIX"

pip install --upgrade pip
pip install vllm
# The full pipeline (uncertainty/eval stages) also runs inside this env:
# sentence-transformers reuses vllm's torch; the rest are lightweight.
pip install sentence-transformers scikit-learn scipy matplotlib

python - <<'PY'
import vllm, torch
print(f"vllm {vllm.__version__}  torch {torch.__version__}")
print(f"cuda available at runtime is expected False on cpu node: {torch.cuda.is_available()}")
PY
echo "[build-vllm-env] DONE exit=$?"
