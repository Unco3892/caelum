#!/usr/bin/env python3
"""Does the lambda BALANCE do anything under the inverse weight geometry?

February-era experiments (pre-judged matching) showed CAELUM with
``weight_fn='inverse'`` beating vanilla APSS at matched coverage, but with
EpistemicOnly_inverse often matching CAELUM_inverse -- i.e. the weighting
helped while the aleatoric/epistemic *balance* lambda did not visibly earn
anything. This script re-tests that at the judged 30-cell grid under the
multiseed headline protocol (40/30/30 split, full-V2 threshold, randomized
APS, alpha=0.1, seeds 42-45):

    TrueAPS          w = 1                       (unweighted APS)
    Vanilla_add      w = 1 + U_ale               (additive, lambda=0)
    Inv_lam0         w = 1/(1 + U_ale)           (inverse, lambda=0)
    Inv_equal        w = 1/(1 + U_ale + U_epi)   (inverse, lambda=1)
    Inv_epionly      w = 1/(1 + U_ale + 1e6*U_epi)  (inverse, lambda->inf)
    Inv_star         inverse, lambda* grid-searched on Val1
    Add_star         additive, lambda* grid-searched on Val1 (reference)

Coverage is reported next to APSS: the conformal quantile targets the same
1 - alpha for every method, so APSS is comparable at (approximately) matched
coverage; cells where coverage differs materially are flagged. Per-item
set-size agreement between Inv_star and Inv_lam0 isolates the lambda balance
from the weight geometry.

Outputs ``results/inverse_multiseed_judged.json``; add ``--write-markdown`` for ``.md``. Never touches
headline files.

Usage:
    python -u src/run_inverse_multiseed.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings as _warnings
from pathlib import Path

import numpy as np

_warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import RESULTS_DIR  # noqa: E402

from caelum.calibration import CAELUMCalibrator  # noqa: E402
from caelum.utils import split_data  # noqa: E402

from run_headline_multiseed import ALPHA, SEEDS, _ensure_types  # noqa: E402
from run_lambda_mechanism import discover_judged_cells  # noqa: E402



def _setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", force=True)
    return logging.getLogger("inverse_multiseed")


def make_calibrator(method: str, seed: int) -> CAELUMCalibrator:
    if method == "TrueAPS":
        return CAELUMCalibrator(alpha=ALPHA, lambda_grid=np.array([0.0]),
                                seed=seed, weight_fn="unit", randomize=True)
    if method == "Vanilla_add":
        return CAELUMCalibrator(alpha=ALPHA, lambda_grid=np.array([0.0]),
                                seed=seed, weight_fn="additive",
                                randomize=True)
    if method == "Inv_lam0":
        return CAELUMCalibrator(alpha=ALPHA, lambda_grid=np.array([0.0]),
                                seed=seed, weight_fn="inverse",
                                randomize=True)
    if method == "Inv_equal":
        return CAELUMCalibrator(alpha=ALPHA, lambda_grid=np.array([1.0]),
                                seed=seed, weight_fn="inverse",
                                randomize=True)
    if method == "Inv_epionly":
        return CAELUMCalibrator(alpha=ALPHA, lambda_grid=np.array([1e6]),
                                seed=seed, weight_fn="inverse",
                                randomize=True)
    if method == "Inv_star":
        return CAELUMCalibrator(alpha=ALPHA, seed=seed, weight_fn="inverse",
                                randomize=True)
    if method == "Add_star":
        return CAELUMCalibrator(alpha=ALPHA, seed=seed, weight_fn="additive",
                                randomize=True)
    raise ValueError(method)


METHODS = ["TrueAPS", "Vanilla_add", "Inv_lam0", "Inv_equal", "Inv_epionly",
           "Inv_star", "Add_star"]


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-markdown", action="store_true",
                        help="Also write the Markdown report.")
    args = parser.parse_args()

    logger = _setup_logging()
    t0 = time.time()
    cells = discover_judged_cells()
    logger.info("inverse multiseed [judged]: %d cells, seeds=%s, alpha=%.2f",
                len(cells), SEEDS, ALPHA)

    out: dict[str, dict] = {}
    for ci, cell in enumerate(cells, 1):
        with open(cell["path"], encoding="utf-8") as f:
            data = _ensure_types(json.load(f))
        acc: dict[str, dict[str, list[float]]] = {
            m: {"apss": [], "cov": [], "cov_matched": [], "lam": []}
            for m in METHODS}
        agree: dict[str, list[float]] = {"star_vs_lam0": [],
                                         "star_vs_epionly": [],
                                         "lam0_vs_vanilla": []}
        for seed in SEEDS:
            val1, val2, test = split_data(
                data, val1_frac=0.4, val2_frac=0.3, seed=seed)
            cidx = np.array(
                [int(it.get("correct_cluster_idx", -1)) for it in test])
            matched = cidx >= 0
            sizes: dict[str, np.ndarray] = {}
            for m in METHODS:
                cal = make_calibrator(m, seed).calibrate(
                    val1, val2, verbose=False)
                s = np.array([cal.predict_set_size(it) for it in test])
                sizes[m] = s
                acc[m]["apss"].append(float(s.mean()))
                acc[m]["cov"].append(float(((cidx >= 0) & (cidx < s)).mean()))
                acc[m]["cov_matched"].append(
                    float((cidx[matched] < s[matched]).mean())
                    if matched.any() else float("nan"))
                acc[m]["lam"].append(float(cal.optimal_lambda))
            agree["star_vs_lam0"].append(
                float((sizes["Inv_star"] != sizes["Inv_lam0"]).mean()))
            agree["star_vs_epionly"].append(
                float((sizes["Inv_star"] != sizes["Inv_epionly"]).mean()))
            agree["lam0_vs_vanilla"].append(
                float((sizes["Inv_lam0"] != sizes["Vanilla_add"]).mean()))

        rec = {
            "n": len(data),
            "methods": {
                m: {k: [float(np.nanmean(v)), float(np.nanstd(v))]
                    for k, v in d.items()} for m, d in acc.items()},
            "frac_size_diff": {
                k: [float(np.mean(v)), float(np.std(v))]
                for k, v in agree.items()},
        }
        out[cell["cell_key"]] = rec
        ms = rec["methods"]
        logger.info(
            "[%d/%d] %s: APSS van=%.2f inv0=%.2f inv*=%.2f (lam*=%.2g) "
            "invEpi=%.2f | cov van=%.2f inv*=%.2f | diff(inv*,inv0)=%.3f",
            ci, len(cells), cell["cell_key"],
            ms["Vanilla_add"]["apss"][0], ms["Inv_lam0"]["apss"][0],
            ms["Inv_star"]["apss"][0], ms["Inv_star"]["lam"][0],
            ms["Inv_epionly"]["apss"][0],
            ms["Vanilla_add"]["cov"][0], ms["Inv_star"]["cov"][0],
            rec["frac_size_diff"]["star_vs_lam0"][0],
        )

    # aggregates
    agg = {}
    for m in METHODS:
        agg[m] = {
            "apss": float(np.mean([out[k]["methods"][m]["apss"][0]
                                   for k in out])),
            "cov": float(np.mean([out[k]["methods"][m]["cov"][0]
                                  for k in out])),
            "cov_matched": float(np.nanmean(
                [out[k]["methods"][m]["cov_matched"][0] for k in out])),
        }
    agg["frac_size_diff"] = {
        k: float(np.mean([out[c]["frac_size_diff"][k][0] for c in out]))
        for k in ("star_vs_lam0", "star_vs_epionly", "lam0_vs_vanilla")}
    logger.info("AGGREGATE: %s", json.dumps(agg, indent=1))

    with open(RESULTS_DIR / "inverse_multiseed_judged.json", "w",
              encoding="utf-8") as f:
        json.dump({"alpha": ALPHA, "seeds": SEEDS, "methods": METHODS,
                   "aggregate": agg, "cells": out}, f, indent=1,
                  default=float)

    lines = ["# Inverse-weight multiseed (judged caches, headline protocol)",
             "", "APSS (mean over seeds); coverage in parentheses.", "",
             "| Cell | " + " | ".join(METHODS) + " | diff(inv*,inv0) |",
             "|---|" + "---|" * (len(METHODS) + 1)]
    for key, rec in out.items():
        ms = rec["methods"]
        row = [f"{ms[m]['apss'][0]:.2f} ({ms[m]['cov'][0]:.2f})"
               for m in METHODS]
        lines.append(f"| {key} | " + " | ".join(row)
                     + f" | {rec['frac_size_diff']['star_vs_lam0'][0]:.3f} |")
    lines += ["", "## Aggregate", "", "```json",
              json.dumps(agg, indent=1), "```"]
    if args.write_markdown:
        with open(RESULTS_DIR / "inverse_multiseed_judged.md", "w",
                  encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    logger.info("done in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
