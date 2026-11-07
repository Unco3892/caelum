#!/bin/bash --login
# =============================================================================
# Prefetch Qwen3.6 once into the shared scratch Hugging Face cache.
#
# This prevents concurrent array tasks from racing while downloading the same
# large model files.
#
# Submit:
#   sbatch slurm/prefetch_qwen36.sh
# =============================================================================
#SBATCH --job-name prefetch-q36
#SBATCH --partition cpu
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --time 04:00:00
#SBATCH --output /scratch/%u/caelum/logs/prefetch_q36_%j.out
#SBATCH --error  /scratch/%u/caelum/logs/prefetch_q36_%j.err

WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT to the project directory on the cluster (see slurm/README.md)}"
ENV_PREFIX="${ENV_PREFIX:-$WORK_ROOT/envs/caelum}"
SCRATCH="${SCRATCH:-/scratch/$USER}"
LOG_DIR="$SCRATCH/caelum/logs"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
export MODEL

for p in /etc/profile.d/lmodstacks.sh /etc/profile.d/dcsr_path.sh; do [ -f "$p" ] && source "$p"; done
module purge
dcsrsoft use 20241118
module load miniforge3/24.11.3-2
conda_init
conda activate "$ENV_PREFIX"

if [ -f "$WORK_ROOT/.env" ]; then
    set -a; source "$WORK_ROOT/.env"; set +a
fi
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN

mkdir -p "$LOG_DIR"
export HF_HOME="$SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$SCRATCH/huggingface"
export HF_HUB_DISABLE_PROGRESS_BARS=1

python - <<'PY'
import os
from huggingface_hub import snapshot_download

model = os.environ.get("MODEL", "Qwen/Qwen3.6-35B-A3B")
print(f"Prefetching {model}")
path = snapshot_download(
    repo_id=model,
    resume_download=True,
    max_workers=1,
)
print(f"Cached at {path}")
PY
