#!/bin/bash --login
# =============================================================================
# Cache-only support and abstention analysis for the frontier answer-model pilot.
# =============================================================================
#SBATCH --job-name caelum-front-post
#SBATCH --partition cpu
#SBATCH --cpus-per-task 8
#SBATCH --mem 48G
#SBATCH --time 02:00:00
#SBATCH --output /scratch/%u/caelum/logs/frontier_post_%j.out
#SBATCH --error  /scratch/%u/caelum/logs/frontier_post_%j.err

WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT to the project directory on the cluster (see slurm/README.md)}"
ENV_PREFIX="${ENV_PREFIX:-$WORK_ROOT/envs/caelum}"
SCRATCH="${SCRATCH:-/scratch/$USER}"
CAELUM_DIR="$SCRATCH/caelum_frontier_answer_pilot"
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
LOG_DIR="$SCRATCH/caelum/logs"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

for p in /etc/profile.d/lmodstacks.sh /etc/profile.d/dcsr_path.sh; do [ -f "$p" ] && source "$p"; done
module purge
dcsrsoft use 20241118
module load miniforge3/24.11.3-2
conda_init
conda activate "$ENV_PREFIX"

export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1

cd "$CAELUM_DIR"

echo "[frontier-post] Available uncertainty caches:"
find "$RESULTS_DIR/generations" -maxdepth 1 -type f -name 'unc_*_vne_*.json' -printf '%f\n' | sort

python -u src/run_support_abstain_trial.py --generations_dir "$RESULTS_DIR/generations" --output_dir "$RESULTS_DIR"

PROJ_RESULTS="$WORK_ROOT/caelum_frontier_answer_pilot/results"
mkdir -p "$PROJ_RESULTS"
rsync -a "$RESULTS_DIR/" "$PROJ_RESULTS/"
echo "[frontier-post] Results synced to $PROJ_RESULTS"
