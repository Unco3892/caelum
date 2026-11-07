#!/usr/bin/env python3
"""R(x)-driven selective-answering experiment.

Evaluates R(x) by simulating an escalation policy:
when R(x) > threshold, escalate to an oracle (perfect answer); otherwise use
the model's top-cluster prediction. We compare five policies:

  - R(x)-driven    : escalate items ranked by R(x) descending
  - U_epi-driven   : escalate items ranked by U_epi (CAELUM internal)
  - U_total-driven : escalate items ranked by U_total (Spectral baseline)
  - Random         : random escalation (reference policy)
  - Oracle         : escalate the actually-incorrect items first (upper bound)

Metric: accuracy as a function of "escalation rate" (fraction of items handed
to the oracle). The area between R(x) and Random measures the actionable value
of CAELUM's diagnostic — how much R(x) saves over a no-information baseline.

Output:
  results/r_protocol_curves.png   (per-cell risk-coverage curves)
  results/r_protocol_summary.md (optional; --write-markdown or --output_md)

Usage:
    python src/r_protocol_experiment.py
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, PLOTS_DIR, RESULTS_DIR  # noqa: E402

from caelum.plotting import apply_paper_style
apply_paper_style()

from caelum.calibration import CAELUMCalibrator
from caelum.utils import split_data


SKIP_DATASETS = {"condambigqa"}
SKIP_NSAMPLE_RE = re.compile(r"_50\.json$")

# The accepted 18-cell benchmark uses these three LLMs. Larger-model caches
# (qwen3_14b, qwen3_32b) share results, so they are skipped unless --models
# names them explicitly.
ACCEPTED_MODELS = ("qwen3_8b", "ministral_3_8b_instruct_2512", "gemma_4_e4b_it")
UNC_RE = re.compile(r"^unc_([a-z0-9]+)_(.+)_vne_(\d+)\.json$")

MODEL_PRETTY = {
    "qwen3_8b": "Qwen3-8B",
    "ministral_3_8b_instruct_2512": "Ministral-3-8B",
    "gemma_4_e4b_it": "Gemma-4-E4B",
}
DATASET_PRETTY = {
    "triviaqa": "TriviaQA", "ambigqa": "AmbigQA", "opennq": "OpenNQ",
    "ambiginst": "AmbigInst", "simpleqa": "SimpleQA", "popqa": "PopQA",
}


def parse_unc(name):
    m = UNC_RE.match(name)
    if not m:
        return None
    return {"dataset": m.group(1), "model": m.group(2), "n_samples": int(m.group(3))}


def base_accuracy_per_item(items):
    """Return 1 if model's top cluster matches GT, else 0."""
    return np.array([1 if it.get("correct_cluster_idx", -1) == 0 else 0 for it in items])


def policy_accuracy_curve(items, scores, n_steps=21):
    """Sort items by `scores` desc; escalate top-k to oracle (oracle accuracy=1).

    Returns (escalation_rates, accuracies).
    """
    base_acc = base_accuracy_per_item(items)
    # Stable sort: tied scores keep their input order on every NumPy version.
    order = np.argsort(-np.asarray(scores), kind="stable")
    n = len(items)
    rates = np.linspace(0, 1, n_steps)
    accs = []
    for r in rates:
        k = int(round(r * n))
        # The first k (in score-desc order) are escalated -> accuracy 1
        # The rest get base accuracy
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[order[:k]] = False
        # accuracy = (k oracle correct + sum of non-escalated correct) / n
        acc_non_esc = base_acc[keep_mask].sum()
        accs.append((k + acc_non_esc) / n)
    return rates, np.array(accs)


def auc_under_curve(rates, accs):
    """Trapezoid AUC of accuracy vs escalation rate (higher = better policy)."""
    # np.trapz was renamed np.trapezoid in NumPy 2.0 (same computation).
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(accs, rates))


def evaluate_cell(unc_path, alpha=0.1, seed=42):
    with open(unc_path, encoding="utf-8") as f:
        items = json.load(f)
    if len(items) < 30:
        return None
    val1, val2, test = split_data(items, val1_frac=0.4, val2_frac=0.3, seed=seed)

    cal = CAELUMCalibrator(alpha=alpha, seed=seed)
    cal.calibrate(val1, val2, verbose=False)

    # Compute scores on TEST
    r_scores = np.array([cal.diagnostic_ratio(it) for it in test])
    u_total = np.array([it.get("U_total", it["U_ale"] + it["U_epi"]) for it in test])
    u_epi = np.array([it["U_epi"] for it in test])
    u_ale = np.array([it["U_ale"] for it in test])
    base_acc = base_accuracy_per_item(test)
    # Oracle: 1 - base_acc (escalate the wrong ones first)
    incorrect_score = 1 - base_acc + np.random.RandomState(seed).normal(0, 1e-3, len(test))
    rng = np.random.RandomState(seed)
    random_score = rng.uniform(size=len(test))

    rates, acc_R = policy_accuracy_curve(test, r_scores)
    _, acc_Uepi = policy_accuracy_curve(test, u_epi)
    _, acc_Utotal = policy_accuracy_curve(test, u_total)
    _, acc_random = policy_accuracy_curve(test, random_score)
    _, acc_oracle = policy_accuracy_curve(test, incorrect_score)

    return {
        "rates": rates.tolist(),
        "R(x)": acc_R.tolist(),
        "U_epi": acc_Uepi.tolist(),
        "U_total": acc_Utotal.tolist(),
        "Random": acc_random.tolist(),
        "Oracle": acc_oracle.tolist(),
        "auc": {
            "R(x)": auc_under_curve(rates, acc_R),
            "U_epi": auc_under_curve(rates, acc_Uepi),
            "U_total": auc_under_curve(rates, acc_Utotal),
            "Random": auc_under_curve(rates, acc_random),
            "Oracle": auc_under_curve(rates, acc_oracle),
        },
        "base_accuracy": float(base_acc.mean()),
        "lambda_star": cal.optimal_lambda,
        "n_test": len(test),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations_dir", default=GENERATIONS_DIR, type=Path,
                    help="Folder with the unc_* caches.")
    ap.add_argument("--results_dir", default=RESULTS_DIR, type=Path,
                    help="Folder for r_protocol_results.json.")
    ap.add_argument("--plots_dir", default=PLOTS_DIR, type=Path)
    ap.add_argument("--output_md", default=None,
                    help="Write Markdown to this path (also enables the summary).")
    ap.add_argument("--models", nargs="+", default=list(ACCEPTED_MODELS),
                    help="model keys to include (default: the accepted 18-cell set)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing results file")
    ap.add_argument("--write-markdown", action="store_true",
                    help="Write an optional Markdown summary.")
    args = ap.parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log = logging.getLogger("r_protocol")

    results_json = args.results_dir / "r_protocol_results.json"
    if results_json.exists() and not args.overwrite:
        raise SystemExit(f"{results_json} already exists and backs reported results; "
                         "pass --overwrite or write elsewhere.")
    cells = []  # list of (model, dataset, metrics)
    for f in sorted(args.generations_dir.glob("unc_*.json")):
        if SKIP_NSAMPLE_RE.search(f.name):
            continue
        info = parse_unc(f.name)
        if (info is None or info["dataset"] in SKIP_DATASETS
                or info["model"] not in args.models):
            continue
        log.info("Evaluating %s", f.name)
        m = evaluate_cell(f)
        if m:
            cells.append((info["model"], info["dataset"], m))

    # ---- Plot grid ----
    n = len(cells)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 2.5*rows), sharey=True)
    axes = axes.flatten()
    methods_styles = [
        ("Oracle", "#10b981", "-"),
        ("R(x)", "#3b82f6", "-"),
        ("U_epi", "#7c3aed", "--"),
        ("U_total", "#fbbf24", "--"),
        ("Random", "#9aa2ad", ":"),
    ]
    for ax, (m, d, met) in zip(axes, cells):
        for label, color, ls in methods_styles:
            ax.plot(met["rates"], met[label], ls, color=color, label=label, linewidth=1.5)
        ax.set_title(f"{MODEL_PRETTY.get(m,m).split('-')[0]} / {DATASET_PRETTY.get(d,d)}", fontsize=8)
        ax.set_xlabel("Escalation rate", fontsize=8)
        ax.set_ylabel("Accuracy", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    for ax in axes[len(cells):]:
        ax.set_visible(False)
    # one legend on the last visible subplot
    axes[len(cells)-1].legend(loc="lower right", fontsize=7)
    plt.suptitle("R(x)-driven selective answering: accuracy vs escalation rate (each cell)",
                 y=1.0)
    plt.tight_layout()
    plt.savefig(args.plots_dir / "r_protocol_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", args.plots_dir / "r_protocol_curves.png")

    # ---- Markdown summary ----
    lines = ["# R(x)-driven selective answering: AUC of accuracy vs escalation rate\n"]
    lines.append(
        "Each row reports the AUC under the accuracy-vs-escalation-rate curve for "
        "5 escalation policies. **Higher is better.** Oracle is the upper bound "
        "(always escalate the actually-wrong items first). Random is a reference policy. "
        "R(x), U_epi, U_total are 'rank items by this score and escalate top-k'.\n"
    )
    lines.append(
        "**R(x) AUC > Random AUC** means R(x) is informative about which items to escalate. "
        "**R(x) AUC ≈ Oracle AUC** means R(x) is near-optimal.\n"
    )
    headers = ["Model", "Dataset", "n_test", "Base acc", "λ*",
               "AUC<sub>Random</sub>", "AUC<sub>U_total</sub>", "AUC<sub>U_epi</sub>",
               "AUC<sub>R(x)</sub>", "AUC<sub>Oracle</sub>",
               "Δ(R(x) − Random)"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for m, d, met in cells:
        au = met["auc"]
        delta_r = au["R(x)"] - au["Random"]
        lines.append(
            f"| {MODEL_PRETTY.get(m,m)} | {DATASET_PRETTY.get(d,d)} | {met['n_test']} | "
            f"{met['base_accuracy']:.3f} | {met['lambda_star']:.3g} | "
            f"{au['Random']:.3f} | {au['U_total']:.3f} | {au['U_epi']:.3f} | "
            f"{au['R(x)']:.3f} | {au['Oracle']:.3f} | {delta_r:+.3f} |"
        )
    if args.write_markdown or args.output_md is not None:
        output_md = Path(args.output_md or "results/r_protocol_summary.md")
        output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info("Wrote: %s", output_md)

    # ---- JSON dump ----
    out = {f"{m}__{d}": met for m, d, met in cells}
    Path(args.results_dir / "r_protocol_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
