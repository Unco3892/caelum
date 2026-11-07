#!/usr/bin/env python3
"""Operating-point audit of the pre-generation matchability classifier.

Independent reproduction of the classifier operating-point audit from the rebuttal.
The question it answers is Reviewer ``g1w9`` Q4: how good is the *learned*
pre-generation matchability classifier m_hat(x) on its own terms, without any
oracle fallback, and what does its threshold tradeoff look like?

Protocol (replicated exactly from the audited file):
    - 30 judged-label (model, dataset) cells: five generators (Qwen3-8B,
      Ministral-3-8B, Gemma-4-E4B-it, Qwen3-14B, Qwen3-32B) x six datasets
      (TriviaQA, AmbigQA, OpenNQ, AmbigInst, SimpleQA, PopQA). Labels come from
      the LLM-judged caches in ``results/llm_judged_unc_full``.
    - Seeds 42-45, giving 30 x 4 = 120 cell-seed folds.
    - Per fold a 30 / 30 / 40 split: V0 (classifier training), V2
      (calibration / diagnostic, unused by this audit), Test (evaluation).
      The permutation follows ``caelum.utils.split_data`` exactly.
    - Features: frozen ``all-mpnet-base-v2`` question embeddings plus the
      paper's handcrafted pre-generation features
      (:func:`caelum.abstention.build_pregen_features`).
    - Model: L2 logistic regression, ``C=0.1``, balanced class weights
      (:func:`caelum.abstention.train_matchability_classifier`).
    - Positive class is MATCHABLE (``correct_cluster_idx >= 0``). Note this is
      the OPPOSITE polarity from the detection tables, which score support
      failure.
    - No oracle substitution: a single-class V0 fold falls back to that fold's
      constant base-rate prediction.
    - Macro metrics weight each cell-seed fold equally; folds where a metric is
      undefined (single-class test labels, or no selected item at a threshold)
      are dropped from that metric's average.
    - ECE uses 10 equal-width probability bins.

Outputs (under ``results``):
    - operating_point_audit_repro.json  (per-fold rows + macro summary)
    - operating_point_audit.txt         (run log)

Usage::

    python -u src/run_operating_point_audit.py

    # Smoke test on a couple of cells
    python -u src/run_operating_point_audit.py \
        --cells qwen3_8b__triviaqa qwen3_8b__popqa
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings as _warnings
from pathlib import Path

import numpy as np

_warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, JUDGED_DIR, RESULTS_DIR, repo_relative  # noqa: E402

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from caelum.abstention import (  # noqa: E402
    build_pregen_features,
    train_matchability_classifier,
)
from caelum.utils import SentenceEncoder, split_data  # noqa: E402
from run_judged_rerun import (  # noqa: E402
    DATASET_PRETTY,
    _ensure_types,
    _setup_logging,
    discover_cells,
    resolve_source,
)

LOG_PATH = RESULTS_DIR / "operating_point_audit.txt"
OUT_JSON = RESULTS_DIR / "operating_point_audit_repro.json"

SEEDS = [42, 43, 44, 45]
V0_FRAC = 0.3
V2_FRAC = 0.3
THRESHOLDS = (0.3, 0.5, 0.7)
N_ECE_BINS = 10
LOGISTIC_C = 0.1

# Dataset whose pooled block the audited file reports separately.
POOLED_DATASET = "popqa"


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

class CachedSentenceEncoder:
    """Memoising wrapper around :class:`caelum.utils.SentenceEncoder`.

    Every question in the 30-cell pool is embedded exactly once, so the
    training and test feature matrices are guaranteed bit-identical for a
    repeated question and the audit is not sensitive to batch composition.

    Attributes:
        device: Device string passed to the underlying encoder.
        n_encoded: Number of distinct texts embedded so far.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2",
                 device: str | None = None) -> None:
        self._encoder = SentenceEncoder(model_name=model_name, device=device)
        self._cache: dict[str, np.ndarray] = {}
        self.device = self._encoder.device
        self.n_encoded = 0

    def warm(self, texts: list[str], batch_size: int = 64) -> None:
        """Pre-embed every distinct text in ``texts`` in one deterministic pass."""
        todo = sorted({t for t in texts if t not in self._cache})
        if not todo:
            return
        embs = self._encoder.encode(todo, batch_size=batch_size, normalize=True)
        for text, emb in zip(todo, np.asarray(embs)):
            self._cache[text] = np.asarray(emb, dtype=float)
        self.n_encoded += len(todo)

    def encode(self, texts: list[str], batch_size: int = 64,
               normalize: bool = True) -> np.ndarray:
        """Return cached (or freshly computed) normalised embeddings."""
        missing = [t for t in texts if t not in self._cache]
        if missing:
            self.warm(missing, batch_size=batch_size)
        if not texts:
            return np.zeros((0, 768), dtype=float)
        return np.stack([self._cache[t] for t in texts], axis=0)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_indices(
    n: int, seed: int, v0_frac: float = V0_FRAC, v2_frac: float = V2_FRAC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index-level replica of :func:`caelum.utils.split_data`.

    Args:
        n: Number of items in the cell.
        seed: Seed for ``np.random.RandomState``.
        v0_frac: Fraction assigned to the classifier-training split.
        v2_frac: Fraction assigned to the calibration/diagnostic split.

    Returns:
        ``(v0_idx, v2_idx, test_idx)`` index arrays into the cell's item list.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_v0 = int(n * v0_frac)
    n_v2 = int(n * v2_frac)
    return (indices[:n_v0], indices[n_v0:n_v0 + n_v2], indices[n_v0 + n_v2:])


def _assert_split_matches_canonical(data: list[dict], seed: int) -> None:
    """Fail loudly if :func:`split_indices` diverges from ``split_data``."""
    v0_idx, v2_idx, test_idx = split_indices(len(data), seed)
    v0, v2, test = split_data(
        data, val1_frac=V0_FRAC, val2_frac=V2_FRAC, seed=seed,
    )
    for idx, ref, name in (
        (v0_idx, v0, "V0"), (v2_idx, v2, "V2"), (test_idx, test, "Test"),
    ):
        assert len(idx) == len(ref), f"{name} size mismatch"
        assert all(data[i] is r for i, r in zip(idx, ref)), \
            f"{name} membership diverges from caelum.utils.split_data"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = N_ECE_BINS,
) -> float:
    """Expected calibration error with equal-width probability bins.

    ECE = sum_b (n_b / n) * |mean(y in bin b) - mean(p in bin b)|, with bins
    partitioning [0, 1] into ``n_bins`` equal-width intervals (p = 1.0 falls in
    the top bin).

    Args:
        y_true: Binary labels (1 = matchable).
        probs: Predicted P(matchable).
        n_bins: Number of equal-width bins.

    Returns:
        The ECE, or NaN when there are no items.
    """
    n = len(y_true)
    if n == 0:
        return float("nan")
    bin_idx = np.clip((np.asarray(probs) * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        ece += (n_b / n) * abs(
            float(y_true[mask].mean()) - float(np.asarray(probs)[mask].mean())
        )
    return float(ece)


def threshold_metrics(
    y_true: np.ndarray, probs: np.ndarray, threshold: float,
) -> dict[str, float]:
    """Selection metrics at one operating point on P(matchable).

    An item is *selected* (predicted matchable, i.e. answer rather than
    abstain) when ``P(matchable) >= threshold``.

    Args:
        y_true: Binary labels (1 = matchable).
        probs: Predicted P(matchable).
        threshold: Operating point.

    Returns:
        Dict with ``selected_fraction``, ``precision``, ``recall``, ``rho``
        (= 1 - precision) and ``n_selected``. Precision / rho are NaN when no
        item is selected; recall is NaN when the fold has no matchable item.
    """
    n = len(y_true)
    if n == 0:
        return {"selected_fraction": float("nan"), "precision": float("nan"),
                "recall": float("nan"), "rho": float("nan"), "n_selected": 0}
    selected = np.asarray(probs) >= threshold
    n_selected = int(selected.sum())
    n_pos = int((y_true == 1).sum())
    precision = float(y_true[selected].mean()) if n_selected > 0 else float("nan")
    recall = (
        float(selected[y_true == 1].mean()) if n_pos > 0 else float("nan")
    )
    return {
        "selected_fraction": float(selected.mean()),
        "precision": precision,
        "recall": recall,
        "rho": (1.0 - precision) if n_selected > 0 else float("nan"),
        "n_selected": n_selected,
    }


def discrimination_metrics(
    y_true: np.ndarray, probs: np.ndarray,
) -> dict[str, float]:
    """AUROC / average precision / Brier / ECE for one set of predictions."""
    n = len(y_true)
    if n == 0:
        return {"auroc": float("nan"), "average_precision": float("nan"),
                "brier": float("nan"), "ece": float("nan")}
    both_classes = len(np.unique(y_true)) == 2
    auroc = float(roc_auc_score(y_true, probs)) if both_classes else float("nan")
    ap = (
        float(average_precision_score(y_true, probs))
        if int((y_true == 1).sum()) > 0 else float("nan")
    )
    brier = float(np.mean((np.asarray(probs) - y_true) ** 2))
    return {
        "auroc": auroc, "average_precision": ap, "brier": brier,
        "ece": expected_calibration_error(y_true, probs),
    }


def _mean_sd(values: list[float]) -> tuple[float, float, float, int]:
    """Macro mean and both SD conventions over the finite entries.

    Returns:
        ``(mean, sd_population, sd_sample, n_defined)``. Both SD conventions are
        reported because the audited file does not state which it used, and the
        two differ in the third decimal at n ~ 120.
    """
    arr = np.array([v for v in values if v is not None and np.isfinite(v)],
                   dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    sd_sample = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
    return float(arr.mean()), float(arr.std()), sd_sample, int(arr.size)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def evaluate_cell(
    data: list[dict], seeds: list[int], encoder: CachedSentenceEncoder,
    cell_key: str, model: str, dataset: str,
) -> list[dict]:
    """Fit and score the classifier on every seed fold of one cell.

    Args:
        data: The cell's judged records.
        seeds: Seeds defining the folds.
        encoder: Shared cached sentence encoder.
        cell_key: ``{model}__{dataset}`` identifier.
        model: Model tag.
        dataset: Dataset tag.

    Returns:
        One record per seed with the fold's metrics, sizes and per-item
        predictions summary.
    """
    X, y, _ = build_pregen_features(data, encoder)
    rows: list[dict] = []
    for seed in seeds:
        v0_idx, v2_idx, test_idx = split_indices(len(data), seed)
        items_v0 = [data[i] for i in v0_idx]
        clf = train_matchability_classifier(
            items_v0, encoder=encoder, seed=seed, C=LOGISTIC_C,
        )
        single_class_v0 = len(np.unique(y[v0_idx])) < 2
        probs_test = np.asarray(clf.predict_proba(X[test_idx])[:, 1], dtype=float)
        y_test = y[test_idx]

        row = {
            "cell_key": cell_key, "model": model, "dataset": dataset,
            "seed": seed, "n_cell": len(data),
            "n_v0": int(len(v0_idx)), "n_v2": int(len(v2_idx)),
            "n_test": int(len(test_idx)),
            "v0_base_rate": float(y[v0_idx].mean()) if len(v0_idx) else float("nan"),
            "test_base_rate": float(y_test.mean()) if len(y_test) else float("nan"),
            "single_class_v0": bool(single_class_v0),
            "single_class_test": bool(len(np.unique(y_test)) < 2),
        }
        row.update(discrimination_metrics(y_test, probs_test))
        for t in THRESHOLDS:
            tag = f"t{int(round(t * 100)):02d}"
            for k, v in threshold_metrics(y_test, probs_test, t).items():
                row[f"{tag}_{k}"] = v
        row["_probs"] = probs_test.tolist()
        row["_labels"] = y_test.astype(int).tolist()
        rows.append(row)
    return rows


def macro_summary(rows: list[dict]) -> dict:
    """Macro-average the per-fold metrics, weighting each fold equally.

    Every metric is averaged over the folds where it is defined. Because
    average precision *is* defined (and equals 1.0) on an all-positive fold
    while AUROC is not, the summary also reports the four discrimination
    metrics restricted to the both-class folds under
    ``both_class_folds_only`` -- the audited file's fold-inclusion rule for
    average precision is ambiguous and the two readings differ.
    """
    out: dict[str, dict] = {}
    for key in ("auroc", "average_precision", "brier", "ece"):
        mu, sd, sd_s, n_def = _mean_sd([r[key] for r in rows])
        out[key] = {"mean": mu, "sd_population": sd, "sd_sample": sd_s,
                    "n_folds_defined": n_def}

    both = [r for r in rows if not r["single_class_test"]]
    out["both_class_folds_only"] = {}
    for key in ("auroc", "average_precision", "brier", "ece"):
        mu, sd, sd_s, n_def = _mean_sd([r[key] for r in both])
        out["both_class_folds_only"][key] = {
            "mean": mu, "sd_population": sd, "sd_sample": sd_s,
            "n_folds_defined": n_def,
        }

    thresholds: dict[str, dict] = {}
    for t in THRESHOLDS:
        tag = f"t{int(round(t * 100)):02d}"
        entry: dict[str, float | int] = {"threshold": t}
        for key in ("selected_fraction", "precision", "recall", "rho"):
            mu, sd, sd_s, n_def = _mean_sd([r[f"{tag}_{key}"] for r in rows])
            entry[f"{key}_mean"] = mu
            entry[f"{key}_sd_population"] = sd
            entry[f"{key}_sd_sample"] = sd_s
            entry[f"{key}_n_folds_defined"] = n_def
        entry["n_folds_with_selection"] = int(sum(
            1 for r in rows if r[f"{tag}_n_selected"] > 0
        ))
        thresholds[tag] = entry
    out["thresholds"] = thresholds
    out["n_folds"] = len(rows)
    out["n_single_class_v0"] = int(sum(1 for r in rows if r["single_class_v0"]))
    out["n_single_class_test"] = int(sum(1 for r in rows if r["single_class_test"]))
    return out


def pooled_summary(rows: list[dict], threshold: float = 0.5) -> dict:
    """Pool every fold's test predictions and score them as one sample."""
    if not rows:
        return {}
    probs = np.concatenate([np.asarray(r["_probs"], dtype=float) for r in rows])
    labels = np.concatenate([np.asarray(r["_labels"], dtype=int) for r in rows])
    out: dict[str, float | int] = {
        "n_items": int(len(labels)), "n_folds": len(rows),
        "threshold": threshold, "base_rate": float(labels.mean()),
    }
    out.update(discrimination_metrics(labels, probs))
    out.update(threshold_metrics(labels, probs, threshold))
    return out


def run_audit(
    judged_dir: Path, base_dir: Path, seeds: list[int],
    cell_filters: list[str] | None, device: str | None, logger,
) -> dict:
    """Execute the full audit and return the JSON-serialisable result dict."""
    cells = discover_cells(base_dir)
    if cell_filters:
        cells = [c for c in cells if any(f in c["cell_key"] for f in cell_filters)]
    if not cells:
        raise SystemExit(f"No cells discovered under {base_dir}")

    resolved = []
    for cell in cells:
        path, source = resolve_source(cell, judged_dir, base_dir)
        resolved.append((cell, path, source))
    n_heuristic = sum(1 for _, _, s in resolved if s != "judged")
    logger.info("Discovered %d cells (%d judged, %d heuristic fallback).",
                len(resolved), len(resolved) - n_heuristic, n_heuristic)

    loaded: list[tuple[dict, list[dict], str]] = []
    all_questions: list[str] = []
    for cell, path, source in resolved:
        with open(path, encoding="utf-8") as f:
            data = _ensure_types(json.load(f))
        loaded.append((cell, data, source))
        all_questions.extend(str(it.get("question", "") or "") for it in data)

    logger.info("Loaded %d items across %d cells (%d distinct questions).",
                len(all_questions), len(loaded), len(set(all_questions)))

    logger.info("Loading all-mpnet-base-v2 (device=%s)...", device or "auto")
    encoder = CachedSentenceEncoder(device=device)
    t_emb = time.time()
    encoder.warm(all_questions)
    logger.info("Embedded %d distinct questions on %s in %.1f s.",
                encoder.n_encoded, encoder.device, time.time() - t_emb)

    # Sanity check: our index split must match the canonical helper exactly.
    _assert_split_matches_canonical(loaded[0][1], seeds[0])

    rows: list[dict] = []
    for i, (cell, data, source) in enumerate(loaded, start=1):
        cell_rows = evaluate_cell(
            data, seeds, encoder, cell["cell_key"], cell["model"], cell["dataset"],
        )
        for r in cell_rows:
            r["source"] = source
        rows.extend(cell_rows)
        aur = [r["auroc"] for r in cell_rows if np.isfinite(r["auroc"])]
        logger.info(
            "[%2d/%d] %-42s n=%4d match=%.3f AUROC=%s",
            i, len(loaded), cell["cell_key"], len(data),
            float(np.mean([int(d["correct_cluster_idx"]) >= 0 for d in data])),
            f"{np.mean(aur):.3f}" if aur else "nan",
        )

    macro = macro_summary(rows)
    pooled = {
        DATASET_PRETTY.get(POOLED_DATASET, POOLED_DATASET): pooled_summary(
            [r for r in rows if r["dataset"] == POOLED_DATASET], 0.5,
        )
    }
    per_cell = {}
    for cell, _, _ in loaded:
        ck = cell["cell_key"]
        per_cell[ck] = macro_summary([r for r in rows if r["cell_key"] == ck])

    # Drop the bulky per-item arrays from the serialised fold rows.
    slim_rows = [{k: v for k, v in r.items() if not k.startswith("_")}
                 for r in rows]
    return {
        "protocol": {
            "judged_dir": repo_relative(judged_dir), "generations_dir": repo_relative(base_dir),
            "seeds": seeds, "v0_frac": V0_FRAC, "v2_frac": V2_FRAC,
            "test_frac": round(1.0 - V0_FRAC - V2_FRAC, 4),
            "logistic_C": LOGISTIC_C, "class_weight": "balanced",
            "positive_class": "matchable (correct_cluster_idx >= 0)",
            "thresholds": list(THRESHOLDS), "n_ece_bins": N_ECE_BINS,
            "encoder": "all-mpnet-base-v2", "device": encoder.device,
            "selection_rule": "P(matchable) >= threshold",
            "sd_convention": "both reported: sd_population (ddof=0) and "
                             "sd_sample (ddof=1), over the defined folds",
        },
        "macro": macro,
        "pooled": pooled,
        "per_cell": per_cell,
        "folds": slim_rows,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def log_report(result: dict, logger) -> None:
    """Print the macro table, threshold table and pooled block to the log."""
    macro = result["macro"]
    logger.info("=" * 72)
    logger.info("MACRO SUMMARY over %d cell-seed folds", macro["n_folds"])
    logger.info("  single-class V0 folds:   %d", macro["n_single_class_v0"])
    logger.info("  single-class test folds: %d", macro["n_single_class_test"])
    logger.info("-" * 72)
    logger.info("%-20s %8s %8s %8s %8s", "metric", "mean", "sd(pop)",
                "sd(samp)", "n_folds")
    for key, label in (
        ("auroc", "AUROC"), ("average_precision", "average precision"),
        ("brier", "Brier score"), ("ece", "ECE (10 bins)"),
    ):
        d = macro[key]
        logger.info("%-20s %8.3f %8.3f %8.3f %8d", label, d["mean"],
                    d["sd_population"], d["sd_sample"], d["n_folds_defined"])
    logger.info("-" * 72)
    logger.info("Same metrics restricted to the both-class test folds:")
    for key, label in (
        ("auroc", "AUROC"), ("average_precision", "average precision"),
        ("brier", "Brier score"), ("ece", "ECE (10 bins)"),
    ):
        d = macro["both_class_folds_only"][key]
        logger.info("%-20s %8.3f %8.3f %8.3f %8d", label, d["mean"],
                    d["sd_population"], d["sd_sample"], d["n_folds_defined"])
    logger.info("-" * 72)
    logger.info("%-10s %10s %10s %10s %10s %8s", "threshold", "selected",
                "precision", "recall", "rho", "n_sel>0")
    for t in THRESHOLDS:
        e = macro["thresholds"][f"t{int(round(t * 100)):02d}"]
        logger.info("%-10.1f %10.3f %10.3f %10.3f %10.3f %8d",
                    t, e["selected_fraction_mean"], e["precision_mean"],
                    e["recall_mean"], e["rho_mean"],
                    e["n_folds_with_selection"])
    for name, block in result["pooled"].items():
        if not block:
            continue
        logger.info("-" * 72)
        logger.info("POOLED %s at threshold %.1f (n=%d items, %d folds)",
                    name, block["threshold"], block["n_items"], block["n_folds"])
        logger.info("  selected fraction %.3f | precision %.3f | recall %.3f "
                    "| rho %.3f", block["selected_fraction"], block["precision"],
                    block["recall"], block["rho"])
        logger.info("  AUROC %.3f | Brier %.3f | ECE %.3f",
                    block["auroc"], block["brier"], block["ece"])
    logger.info("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judged_dir", type=str, default=str(JUDGED_DIR),
                        help="Directory of LLM-judged unc_* caches.")
    parser.add_argument("--generations_dir", type=str, default=str(GENERATIONS_DIR),
                        help="Heuristic cache directory (cell discovery + fallback).")
    parser.add_argument("--cells", type=str, nargs="*", default=None,
                        help="Optional cell_key substring filters.")
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device for the sentence encoder.")
    parser.add_argument("--out", type=str, default=str(OUT_JSON))
    args = parser.parse_args()

    logger = _setup_logging(LOG_PATH)
    t0 = time.time()
    logger.info("=" * 72)
    logger.info("Pre-generation classifier operating-point audit")
    logger.info("judged_dir=%s seeds=%s split=%.0f/%.0f/%.0f",
                args.judged_dir, args.seeds, V0_FRAC * 100, V2_FRAC * 100,
                (1 - V0_FRAC - V2_FRAC) * 100)

    result = run_audit(
        judged_dir=Path(args.judged_dir), base_dir=Path(args.generations_dir),
        seeds=list(args.seeds), cell_filters=args.cells, device=args.device,
        logger=logger,
    )
    log_report(result, logger)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    logger.info("Elapsed: %.1f s", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
