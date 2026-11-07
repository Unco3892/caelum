#!/bin/bash
# =============================================================================
# ConU baseline on the 18 frozen Table 1 cells (one array task per cell).
#
# Stage 1 recovers answer-to-cluster labels and computes the authors' STS
# inputs (GPU) from results/generations/ in the checkout and writes
# results/conu_table1/stage1/<cell>/. It refuses to overwrite a cell folder, so
# move results/conu_table1/stage1/ aside to recompute it. Stage 2 (Table 1
# protocol) is cheap and runs locally:
#   python src/run_conu_baseline.py table1 --root results/conu_table1
#
# Prerequisite: $STS_CACHE, a Hugging Face cache holding
# cross-encoder/stsb-roberta-large.
#
# Usage: sbatch [--account=<account>] --export=ALL,STS_CACHE=/path/to/hf_cache,BASEPY=/path/to/env/bin/python slurm/conu_table1.sh
# =============================================================================
#SBATCH --job-name=conu-table1
#SBATCH --partition=gpu-rtx,gpu,gpu-l40,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-17
set -euo pipefail
CAELUM_DIR="${CAELUM_DIR:-/scratch/$USER/caelum}"
BASEPY="${BASEPY:?Set BASEPY to the python executable of the CAELUM environment}"
STS_CACHE="${STS_CACHE:?set STS_CACHE to the cross-encoder cache directory}"
for p in /etc/profile.d/lmodstacks.sh /etc/profile.d/dcsr_path.sh; do [ -f "$p" ] && source "$p"; done
module load cuda/12.6.2 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_OFFLINE=1
MODELS=(qwen3_8b ministral_3_8b_instruct_2512 gemma_4_e4b_it)
DATASETS=(triviaqa:300 ambigqa:200 opennq:300 ambiginst:304 simpleqa:500 popqa:500)
M=${MODELS[$((SLURM_ARRAY_TASK_ID / 6))]}
D=${DATASETS[$((SLURM_ARRAY_TASK_ID % 6))]}
CELL="unc_${D%%:*}_${M}_vne_${D##*:}"
"$BASEPY" "$CAELUM_DIR/src/run_conu_baseline.py" similarities \
  --input "$CAELUM_DIR/results/generations/$CELL.json" \
  --out "$CAELUM_DIR/results/conu_table1/stage1/$CELL" --cache "$STS_CACHE"
