"""Repository folders for cached generations and analysis results.

The cached answers and uncertainty estimates for each model/dataset cell
(``gen_*.json``, ``unc_*.json``) and the logs of the runs that produced them
(``pipeline_*.txt``) live in ``results/generations/``. The same caches with
LLM-judged matching labels live in ``results/llm_judged_unc_full/``. Analysis
scripts read these and write their JSON outputs to ``results/``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
GENERATIONS_DIR = RESULTS_DIR / "generations"
JUDGED_DIR = RESULTS_DIR / "llm_judged_unc_full"
PLOTS_DIR = RESULTS_DIR / "plots"


def content_sha256(path: str | Path) -> str:
    """SHA-256 of a text file's content in its LF form.

    Git stores text files with LF line endings and Windows checkouts may write
    CRLF, so hashing the LF form gives the same digest on every checkout.
    """
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def repo_relative(path: str | Path) -> str:
    """Return ``path`` relative to the repository, for recording in outputs."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()
