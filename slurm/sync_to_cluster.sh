#!/bin/bash
# =============================================================================
# CAELUM — Sync local code, dependency files and data to cluster scratch
#
# Uses tar-over-ssh to avoid rsync's poor CRLF handling on Windows hosts.
# Run from WSL or Git Bash on Windows.
#
# Usage: REMOTE=user@login-host DEST_BASE=/scratch/<user>/caelum WORK_ROOT=/path/to/project \
#        bash slurm/sync_to_cluster.sh
# =============================================================================
set -euo pipefail

REMOTE="${REMOTE:?Set REMOTE to user@cluster-login-host}"
DEST_BASE="${DEST_BASE:?Set DEST_BASE to the checkout directory on the cluster, e.g. /scratch/<user>/caelum}"
WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT to the project directory on the cluster (see slurm/README.md)}"
SRC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[sync-to] target: $REMOTE:$DEST_BASE"

# ---------------------------------------------------------------------------
# 1. Ensure remote dirs exist
# ---------------------------------------------------------------------------
ssh "$REMOTE" "mkdir -p $DEST_BASE/src $DEST_BASE/data/eval $DEST_BASE/slurm $DEST_BASE/logs"

# ---------------------------------------------------------------------------
# 2. Code and dependency files - tar over ssh, exclude local source archives.
#    Results and plots live in the top-level results/ folder, which is not synced.
# ---------------------------------------------------------------------------
echo "[sync-to] src/ and requirements files (excluding local source archives)"
( cd "$SRC_ROOT" && \
  tar czf - \
    --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='src/results_archive_pre_clean_method' \
    --exclude='src/plots_archive_pre_clean_method' \
    src/ requirements.txt ) | ssh "$REMOTE" "cd $DEST_BASE && tar xzf -"

# ---------------------------------------------------------------------------
# 3. Evaluation data (data/eval/)
# ---------------------------------------------------------------------------
echo "[sync-to] data/eval/"
( cd "$SRC_ROOT" && tar czf - data/eval/ ) | \
    ssh "$REMOTE" "cd $DEST_BASE && tar xzf -"

# ---------------------------------------------------------------------------
# 4. SLURM scripts, dependency lockfile and .env (for HF_TOKEN)
# ---------------------------------------------------------------------------
echo "[sync-to] slurm/ + .env"
( cd "$SRC_ROOT" && tar czf - slurm/ ) | \
    ssh "$REMOTE" "cd $DEST_BASE && tar xzf -"

# .env goes into both scratch and the work root, where the launchers read it
scp "$SRC_ROOT/.env" "$REMOTE:$DEST_BASE/.env"
ssh "$REMOTE" "mkdir -p $WORK_ROOT && \
               cp $DEST_BASE/.env $WORK_ROOT/.env"

# ---------------------------------------------------------------------------
# 6. Strip CRLF from shell scripts (Windows -> Linux line endings)
# ---------------------------------------------------------------------------
echo "[sync-to] Stripping CRLF from .sh files"
ssh "$REMOTE" "find $DEST_BASE/slurm -name '*.sh' -exec sed -i 's/\r\$//' {} \; && \
               chmod +x $DEST_BASE/slurm/*.sh"

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
echo ""
echo "[sync-to] Done."
ssh "$REMOTE" "ls -la $DEST_BASE/ && du -sh $DEST_BASE/* 2>/dev/null"
echo ""
echo "Next steps on cluster:"
echo "  ssh $REMOTE"
echo "  bash $DEST_BASE/slurm/setup_env.sh         # one-time, ~15 min"
echo "  sbatch $DEST_BASE/slurm/smoke_test.sh      # smoke test, ~30 min"
