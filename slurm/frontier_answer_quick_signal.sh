#!/bin/bash --login
# =============================================================================
# CAELUM — quick signal for frontier answer model
#
# Same model and 100 answers/query as frontier_answer_pilot.sh, but only
# 100 examples per dataset to get an early matchability/abstention signal.
# =============================================================================
#SBATCH --job-name caelum-front-q
#SBATCH --partition gpu-h100,gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --cpus-per-task 12
#SBATCH --mem 120G
#SBATCH --time 06:00:00
#SBATCH --output /scratch/%u/caelum/logs/frontier_quick_%A_%a.out
#SBATCH --error  /scratch/%u/caelum/logs/frontier_quick_%A_%a.err

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

DATASETS=(opennq simpleqa popqa)
MODEL="${FRONTIER_MODEL:-Qwen/Qwen3.5-35B-A3B}"
SEED="${SEED:-42}"
N_SAMPLES="${N_SAMPLES:-100}"
N_CLARIFICATIONS="${N_CLARIFICATIONS:-5}"
N_ANSWERS="${N_ANSWERS:-20}"

TASK_ID=${SLURM_ARRAY_TASK_ID}
if [ "$TASK_ID" -ge "${#DATASETS[@]}" ]; then
    echo "Invalid task id $TASK_ID"
    exit 2
fi
DATASET=${DATASETS[$TASK_ID]}

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
export HF_HOME="$SCRATCH/huggingface"
export TRANSFORMERS_CACHE="$SCRATCH/huggingface"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1
export CAELUM_DISABLE_COMPILE=1

EXTRA_ARGS=""
case "$DATASET" in
    opennq)   EXTRA_ARGS="--opennq_path $CAELUM_DIR/data/eval/OpenNQ.json" ;;
    simpleqa) EXTRA_ARGS="--simpleqa_path $CAELUM_DIR/data/eval/SimpleQA.json" ;;
    popqa)    EXTRA_ARGS="--popqa_path $CAELUM_DIR/data/eval/PopQA.json" ;;
esac

cd "$CAELUM_DIR"

echo "[frontier-quick] model=$MODEL dataset=$DATASET n=$N_SAMPLES seed=$SEED answers=$N_ANSWERS"
python -u src/run_pipeline.py \
    --dataset "$DATASET" \
    --model_name "$MODEL" \
    --n_samples "$N_SAMPLES" \
    --seed "$SEED" \
    --n_clarifications "$N_CLARIFICATIONS" \
    --n_answers "$N_ANSWERS" \
    --uncertainty_method vne \
    --weight_fn additive \
    --backend hf \
    --clarif_provider none \
    $EXTRA_ARGS

EXIT_CODE=$?
PROJ_RESULTS="$WORK_ROOT/caelum_frontier_answer_pilot/results"
mkdir -p "$PROJ_RESULTS"
rsync -a "$RESULTS_DIR/" "$PROJ_RESULTS/"
echo "[frontier-quick] Results synced to $PROJ_RESULTS (exit=$EXIT_CODE)"
exit "$EXIT_CODE"
