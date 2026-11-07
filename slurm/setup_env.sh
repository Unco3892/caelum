#!/bin/bash
# Create a separate environment from the recorded DCSR dependencies.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash slurm/setup_env.sh [--prefix PATH]

Create a new Python 3.11.15 environment with slurm/requirements-dcsr.lock.txt.
The target must not exist. Existing environments are never updated or reused.

Options:
  --prefix PATH  New environment directory (default: $ENV_PREFIX, else $WORK_ROOT/envs/caelum)
  -h, --help     Show this help without creating files or loading modules.

Environment:
  WORK_ROOT       Project storage directory on the cluster (required).
  CONDA_PKGS_DIRS  Package cache for this process (default: $WORK_ROOT/conda_pkgs).

This script does not change ~/.condarc, log in to Hugging Face, or download models.
EOF
}

die() {
    printf '[setup] ERROR: %s\n' "$*" >&2
    exit 1
}

WORK_ROOT="${WORK_ROOT:-}"
ENV_PREFIX="${ENV_PREFIX:-}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --prefix)
            [ "$#" -ge 2 ] && [ -n "$2" ] || die "--prefix requires a path."
            ENV_PREFIX="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *) die "Unknown argument: $1 (use --help)." ;;
    esac
done

[ -n "$WORK_ROOT" ] || die "Set WORK_ROOT to the project directory on the cluster (see slurm/README.md)."
[ -n "$ENV_PREFIX" ] || ENV_PREFIX="$WORK_ROOT/envs/caelum"

# Check the supplied path before resolving it, including dangling symlinks.
if [ -e "$ENV_PREFIX" ] || [ -L "$ENV_PREFIX" ]; then
    die "Target already exists: $ENV_PREFIX. Choose a new --prefix."
fi
ENV_PREFIX="$(realpath -m -- "$ENV_PREFIX")"
if [ -e "$ENV_PREFIX" ] || [ -L "$ENV_PREFIX" ]; then
    die "Target already exists: $ENV_PREFIX. Choose a new --prefix."
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
LOCK_FILE="$SCRIPT_DIR/requirements-dcsr.lock.txt"
[ -f "$LOCK_FILE" ] || die "Missing $LOCK_FILE. Use a complete checkout."

# Prefer an already working Conda. A login shell can define a stale `conda`
# function (for example from a retired DCSR stack) that exists but fails, so
# test that it runs. DCSR module initialization may use unset variables or
# return a nonzero status even when it makes Conda available.
if ! conda --version >/dev/null 2>&1; then
    set +eu
    for profile in /etc/profile.d/lmodstacks.sh /etc/profile.d/dcsr_path.sh; do
        if [ -f "$profile" ]; then
            source "$profile"
        fi
    done
    if command -v dcsrsoft >/dev/null 2>&1; then
        dcsrsoft use 20241118
    fi
    if command -v module >/dev/null 2>&1; then
        module load miniforge3/24.11.3-2
    fi
    if command -v conda_init >/dev/null 2>&1; then
        conda_init
    fi
    set -euo pipefail
fi
conda --version >/dev/null 2>&1 || die "Conda is unavailable. Initialize Conda and rerun."

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$WORK_ROOT/conda_pkgs}"
mkdir -p -- "$(dirname -- "$ENV_PREFIX")"
# Resolve the pip requirements entry relative to the repository root.
cd -- "$REPO_ROOT"
if [ -e "$ENV_PREFIX" ] || [ -L "$ENV_PREFIX" ]; then
    die "Target appeared during setup: $ENV_PREFIX. Choose a new --prefix."
fi
printf '[setup] Creating %s with %s\n' "$ENV_PREFIX" "$LOCK_FILE"
conda create --prefix "$ENV_PREFIX" --override-channels --channel conda-forge \
    python=3.11.15 pip=26.0.1 --yes
conda run --prefix "$ENV_PREFIX" python -m pip install -r "$LOCK_FILE"
conda run --prefix "$ENV_PREFIX" python -m pip check
printf '[setup] Environment created. Activate with: conda activate %q\n' "$ENV_PREFIX"
printf '[setup] Model downloads and GPU smoke tests are separate steps.\n'
