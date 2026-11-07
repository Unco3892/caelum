#!/bin/bash
# =============================================================================
# CAELUM — Pull results back from the cluster
# Uses tar-over-ssh (Windows-safe).
# Run from WSL or Git Bash on Windows.
# Usage: REMOTE=user@login-host DEST_BASE=/scratch/<user>/caelum WORK_ROOT=/path/to/project \
#        bash slurm/sync_from_cluster.sh
# =============================================================================
set -euo pipefail

REMOTE="${REMOTE:?Set REMOTE to user@cluster-login-host}"
WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT to the project directory on the cluster (see slurm/README.md)}"
SCRATCH_BASE="${DEST_BASE:?Set DEST_BASE to the checkout directory on the cluster, e.g. /scratch/<user>/caelum}"
DEST_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$DEST_ROOT/results" "$DEST_ROOT/results/plots"

echo "[sync-from] Pulling results (project-space copy made by jobs at end)"
ssh "$REMOTE" "cd $WORK_ROOT/caelum/results 2>/dev/null && tar czf - . 2>/dev/null" \
    | tar xzf - -C "$DEST_ROOT/results/" 2>/dev/null || \
    echo "[sync-from] No results in $WORK_ROOT/caelum/results yet"

echo "[sync-from] Pulling plots"
ssh "$REMOTE" "cd $SCRATCH_BASE/results/plots 2>/dev/null && tar czf - . 2>/dev/null" \
    | tar xzf - -C "$DEST_ROOT/results/plots/" 2>/dev/null || \
    echo "[sync-from] No plots in scratch yet"

echo "[sync-from] Pulling logs"
mkdir -p "$DEST_ROOT/slurm/logs"
ssh "$REMOTE" "cd $SCRATCH_BASE/logs 2>/dev/null && tar czf - . 2>/dev/null" \
    | tar xzf - -C "$DEST_ROOT/slurm/logs/" 2>/dev/null || \
    echo "[sync-from] No logs yet"

echo ""
echo "[sync-from] Done. Local results:"
ls -la "$DEST_ROOT/results/" 2>/dev/null | head -10
