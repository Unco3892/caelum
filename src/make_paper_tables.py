#!/usr/bin/env python3
"""Recreate the original 18-cell LaTeX result tables from cached JSON sources.

These exports preserve the historical table format and labels. In particular,
the column labeled ConU is the self-consistency proxy, not a full reproduction
of ConU. These are not the current formatted manuscript tables.

Every number in every emitted table is read directly from a JSON file produced
by the experiment scripts. Nothing is hard-coded.

Inputs:
    results/stage3_results.json         - main 18-cell x 6-method benchmark
    results/r_protocol_results.json     - selective answering AUC per cell
    results/embedding_ablation.json     - encoder ablation (3 cells x 4 enc.)

Outputs under results/exported_tables/ (one tabular block per file):
    main_tabular.tex          (Table 1: Cov./APSS)
    auroc_tabular.tex         (Table 2: hallucination AUROC)
    r_protocol_tabular.tex    (Table 3: selective answering)
    embedding_tabular.tex     (Table 4: encoder ablation)

Bold / underline conventions:
    - ``\\textbf{}`` marks the row optimum (min for APSS, max for AUROC). All
      cells tied for the optimum are bolded.
    - ``\\underline{}`` marks the second-distinct value (second-best). When
      three or more cells share that value, no underline is emitted (degenerate
      runner-up cluster).

Usage:
    python src/make_paper_tables.py
"""

from __future__ import annotations

import argparse

# Handle help and invalid arguments before loading analysis dependencies.
if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()

import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

# ----------------------------------------------------------------------------
# Display constants (paper-facing strings only - no numerical values)
# ----------------------------------------------------------------------------

MODEL_ORDER = [
    "qwen3_8b",
    "ministral_3_8b_instruct_2512",
    "gemma_4_e4b_it",
]
DATASET_ORDER = ["triviaqa", "ambigqa", "opennq", "ambiginst", "simpleqa", "popqa"]

MODEL_PRETTY = {
    "qwen3_8b": "Qwen3-8B",
    "ministral_3_8b_instruct_2512": "Ministral-3-8B",
    "gemma_4_e4b_it": "Gemma-4-E4B-it",
    "qwen3_14b": "Qwen3-14B",
    "qwen3_32b": "Qwen3-32B",
    "qwen3_235b_a22b_instruct_2507": "Qwen3-235B-A22B",
    "gpt_oss_120b": "GPT-OSS-120B",
}
DATASET_PRETTY = {
    "triviaqa": "TriviaQA",
    "ambigqa": "AmbigQA",
    "opennq": "OpenNQ",
    "ambiginst": "AmbigInst",
    "simpleqa": "SimpleQA",
    "popqa": "PopQA",
}

METHOD_ORDER = [
    "CAELUM",
    "Vanilla",
    "EqualWeight",
    "EpistemicOnly",
    "ConU",
    "SemanticEntropyCP",
]

ENCODER_ORDER = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "BAAI/bge-large-en-v1.5",
    "intfloat/e5-large-v2",
]
ENCODER_PRETTY = {
    "sentence-transformers/all-mpnet-base-v2": "all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L12-v2": "all-MiniLM-L12-v2",
    "BAAI/bge-large-en-v1.5": "bge-large-en-v1.5",
    "intfloat/e5-large-v2": "e5-large-v2",
}
EMBEDDING_TABLE_FILES = [
    "gen_triviaqa_qwen3_8b_300.json",
    "gen_ambigqa_qwen3_8b_200.json",
    "gen_popqa_qwen3_8b_500.json",
]


# ----------------------------------------------------------------------------
# Number formatting helpers
# ----------------------------------------------------------------------------


def is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def fmt_lambda(value: float) -> str:
    """Lambda* display: 3 significant figures with padded decimals.

    Examples:
        0.0   -> '0.00'
        0.283 -> '0.28'
        9.32  -> '9.32'
        27.0  -> '27.0'
        47.4  -> '47.4'
    """
    if value is None or is_nan(value):
        return "n/a"
    if value == 0:
        return "0.00"
    abs_v = abs(value)
    if abs_v >= 100:
        return f"{value:.0f}"
    if abs_v >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def fmt_signed(value: float, decimals: int = 3) -> str:
    """Signed delta with explicit '+' or LaTeX minus (``$-$``)."""
    if value is None or is_nan(value):
        return "n/a"
    if value >= 0:
        return f"+{value:.{decimals}f}"
    return f"$-${abs(value):.{decimals}f}"


def _decimals_in_fmt(fmt: str) -> int:
    m = re.search(r"\.(\d+)f", fmt)
    return int(m.group(1)) if m else 2


def mark_best(
    values: Sequence,
    fmt: str = "{:.3f}",
    optimum: str = "max",
    skip_tie: int = 3,
    formatter=None,
) -> list[str]:
    """Return list of LaTeX strings with best bolded and second-best underlined.

    Args:
        values: list of float-or-None-or-NaN
        fmt: format string for plain display (used when ``formatter`` is None)
        optimum: 'max' or 'min'
        skip_tie: skip underlining when this many cells (or more) share the
            second-distinct value (degenerate runner-up cluster)
        formatter: optional callable mapping value -> str. If provided, used
            instead of ``fmt``. The formatter must produce identical output for
            equal values; tie detection is done on rounded raw values.

    Cells with None / NaN are rendered as ``n/a`` (no bold / underline).
    """
    decimals = _decimals_in_fmt(fmt)

    def _round(v):
        if v is None or is_nan(v):
            return None
        return round(float(v), decimals)

    def _display(v):
        if v is None or is_nan(v):
            return "n/a"
        return formatter(v) if formatter is not None else fmt.format(v)

    rounded = [_round(v) for v in values]
    valid = [r for r in rounded if r is not None]
    if not valid:
        return [_display(v) for v in values]

    if optimum == "max":
        best = max(valid)
        below = [r for r in valid if r < best]
        second = max(below) if below else None
    elif optimum == "min":
        best = min(valid)
        above = [r for r in valid if r > best]
        second = min(above) if above else None
    else:
        raise ValueError(f"optimum must be 'max' or 'min', got {optimum!r}")

    second_count = sum(1 for r in valid if r == second) if second is not None else 0
    underline = second is not None and second_count < skip_tie

    out: list[str] = []
    for r, v in zip(rounded, values):
        if r is None:
            out.append("n/a")
            continue
        text = _display(v)
        if r == best:
            out.append(r"\textbf{" + text + "}")
        elif underline and r == second:
            out.append(r"\underline{" + text + "}")
        else:
            out.append(text)
    return out


# ----------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
# Export analysis tables to results/exported_tables/ (gitignored).
TABLES_OUT = RESULTS / "exported_tables"


def load_stage3() -> dict:
    return json.loads((RESULTS / "stage3_results.json").read_text(encoding="utf-8"))


def load_r_protocol() -> dict:
    return json.loads((RESULTS / "r_protocol_results.json").read_text(encoding="utf-8"))


def load_embedding() -> dict:
    return json.loads((RESULTS / "embedding_ablation.json").read_text(encoding="utf-8"))


def cell_key(model: str, dataset: str) -> str:
    return f"{model}__{dataset}"


# ----------------------------------------------------------------------------
# Table 1: main coverage / APSS
# ----------------------------------------------------------------------------


def build_table_main(stage3: dict) -> str:
    lines: list[str] = []
    lines.append(r"\begin{tabular}{llrrrcccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{Dataset} & N & $\pi$ & "
        r"Cov$_{\textsc{caelum}}$ & "
        r"\multicolumn{6}{c}{\textbf{APSS} (lower is better)} & "
        r"$\lambdastar$ \\"
    )
    lines.append(r"\cmidrule(lr){6-11}")
    lines.append(
        r"& & & & & CAELUM & Vanilla & Equal & EpiOnly & ConU & SE-CP & \\"
    )
    lines.append(r"\midrule")

    for mi, model in enumerate(MODEL_ORDER):
        if mi > 0:
            lines.append(r"\midrule")
        lines.append(rf"\multirow{{6}}{{*}}{{{MODEL_PRETTY[model]}}}")
        for dataset in DATASET_ORDER:
            cell = stage3[cell_key(model, dataset)]
            n = cell["n_samples"]
            match_rate = cell["match_rate"]
            caelum_cov = cell["methods"]["CAELUM"]["coverage"]
            lambda_star = cell["methods"]["CAELUM"]["lambda_star"]

            apss_values = [cell["methods"][m]["apss"] for m in METHOD_ORDER]
            apss_marked = mark_best(apss_values, fmt="{:.2f}", optimum="min")

            row = (
                f" & {DATASET_PRETTY[dataset]:<9} & {n} & {match_rate:.3f} & "
                f"{caelum_cov:.3f} & "
                + " & ".join(apss_marked)
                + f" & {fmt_lambda(lambda_star)} \\\\"
            )
            lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# Table 2: hallucination AUROC
# ----------------------------------------------------------------------------

AUROC_KEYS = ["U_total", "U_epi", "U_ale", "R(x)"]


def _cell_aurocs(cell: dict) -> tuple[list, bool]:
    """Return (auroc_values, is_degenerate). Degenerate = NaN AUROCs."""
    auc = cell["aurocs"]
    values = [auc[k] for k in AUROC_KEYS]
    degenerate = all(is_nan(v) for v in values)
    return values, degenerate


def build_table_auroc(stage3: dict) -> str:
    lines: list[str] = []
    lines.append(r"\begin{tabular}{llccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{Dataset} & Match\% & "
        r"$\Utot$ & $\Uepi$ & $\Uale$ & $R(x)$ & "
        r"$\Delta_\textsc{epi}$ & $\Delta_\textsc{R}$ \\"
    )
    lines.append(r"\midrule")

    valid_per_cell: list[dict] = []  # for mean rows

    for mi, model in enumerate(MODEL_ORDER):
        if mi > 0:
            lines.append(r"\midrule")
        # Count non-degenerate datasets so the multirow spans the right number
        # of rows (degenerate cells are skipped rather than shown as n/a).
        non_degenerate = [
            d for d in DATASET_ORDER
            if not _cell_aurocs(stage3[cell_key(model, d)])[1]
        ]
        lines.append(rf"\multirow{{{len(non_degenerate)}}}{{*}}{{{MODEL_PRETTY[model]}}}")
        for dataset in DATASET_ORDER:
            cell = stage3[cell_key(model, dataset)]
            match_pct = 100.0 * cell["match_rate"]
            values, degenerate = _cell_aurocs(cell)
            if degenerate:
                # Skip degenerate (perfect-match) cells entirely rather than
                # rendering an "n/a" row that wastes vertical space and visually
                # clutters the table.
                continue
            marked = mark_best(values, fmt="{:.3f}", optimum="max")
            u_total, u_epi, u_ale, r_x = values
            delta_epi = u_epi - u_total
            delta_r = r_x - u_total
            row = (
                f" & {DATASET_PRETTY[dataset]:<9} & {match_pct:.1f} & "
                + " & ".join(marked)
                + f" & {fmt_signed(delta_epi, 3)} & {fmt_signed(delta_r, 3)} \\\\"
            )
            lines.append(row)
            valid_per_cell.append(
                {
                    "dataset": dataset,
                    "u_total": u_total,
                    "u_epi": u_epi,
                    "u_ale": u_ale,
                    "r_x": r_x,
                    "delta_epi": delta_epi,
                    "delta_r": delta_r,
                }
            )

    # Mean rows
    if valid_per_cell:
        lines.append(r"\midrule")
        n_all = len(valid_per_cell)
        means_all = [
            sum(c[k] for c in valid_per_cell) / n_all
            for k in ["u_total", "u_epi", "u_ale", "r_x"]
        ]
        marked_all = mark_best(means_all, fmt="{:.3f}", optimum="max")
        delta_epi_mean = sum(c["delta_epi"] for c in valid_per_cell) / n_all
        delta_r_mean = sum(c["delta_r"] for c in valid_per_cell) / n_all
        lines.append(
            rf"\textbf{{Mean ({n_all} cells)}} & & & "
            + " & ".join(marked_all)
            + rf" & \textbf{{{fmt_signed(delta_epi_mean, 3)}}} & "
            + f"{fmt_signed(delta_r_mean, 3)} \\\\"
        )

        triv_cells = [c for c in valid_per_cell if c["dataset"] == "triviaqa"]
        if triv_cells:
            n_t = len(triv_cells)
            means_t = [
                sum(c[k] for c in triv_cells) / n_t
                for k in ["u_total", "u_epi", "u_ale", "r_x"]
            ]
            marked_t = mark_best(means_t, fmt="{:.3f}", optimum="max")
            delta_epi_t = sum(c["delta_epi"] for c in triv_cells) / n_t
            delta_r_t = sum(c["delta_r"] for c in triv_cells) / n_t
            lines.append(
                rf"\textbf{{TriviaQA mean ({n_t})}} & & & "
                + " & ".join(marked_t)
                + f" & {fmt_signed(delta_epi_t, 3)} & "
                + f"{fmt_signed(delta_r_t, 3)} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# Table 3: selective answering (R(x)-driven)
# ----------------------------------------------------------------------------


def build_table_r_protocol(r_proto: dict) -> str:
    """Mean over the three models per dataset.

    For each dataset we compute the mean of the AUC of accuracy-vs-escalation
    rate over the available models. Bold / underline are applied among
    {U_total, U_epi, R(x)} only (Random and Oracle are bounds, not methods).
    """
    lines: list[str] = []
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Dataset} & Random & $\Utot$ & $\Uepi$ & $R(x)$ & "
        r"Oracle & $\Delta_{R(x)}$ \\"
    )
    lines.append(r"\midrule")

    delta_per_dataset: list[float] = []
    delta_rows: list[tuple[str, float, float, float, float, float]] = []

    for dataset in DATASET_ORDER:
        cell_keys_for_d = [cell_key(m, dataset) for m in MODEL_ORDER]
        present = [k for k in cell_keys_for_d if k in r_proto]
        if not present:
            continue
        n = len(present)
        aucs = {
            label: sum(r_proto[k]["auc"][label] for k in present) / n
            for label in ["Random", "U_total", "U_epi", "R(x)", "Oracle"]
        }
        delta_rx = aucs["R(x)"] - aucs["Random"]
        delta_per_dataset.append(delta_rx)
        delta_rows.append(
            (
                dataset,
                aucs["Random"],
                aucs["U_total"],
                aucs["U_epi"],
                aucs["R(x)"],
                aucs["Oracle"],
                delta_rx,
            )
        )

    # First pass: compute method-best/2nd among {U_total, U_epi, R(x)} per row
    # Second pass: compute delta best/2nd across rows
    delta_marks = mark_best(
        [row[6] for row in delta_rows], fmt="{:.3f}", optimum="max", formatter=lambda v: fmt_signed(v, 3)
    )

    for marks_delta, row in zip(delta_marks, delta_rows):
        dataset, rand, u_t, u_e, r_x, oracle, _ = row
        method_marked = mark_best(
            [u_t, u_e, r_x], fmt="{:.3f}", optimum="max"
        )
        line = (
            f"{DATASET_PRETTY[dataset]:<10} & {rand:.3f} & "
            + " & ".join(method_marked)
            + f" & {oracle:.3f} & {marks_delta} \\\\"
        )
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# Table 4: embedding ablation (appendix)
# ----------------------------------------------------------------------------


def _dataset_from_filename(filename: str) -> str:
    # gen_<dataset>_<model>_<n>.json -> <dataset>
    parts = filename.replace("gen_", "").split("_")
    return parts[0]


def build_table_embedding(emb: dict) -> str:
    lines: list[str] = []
    lines.append(r"\begin{tabular}{llrrcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Dataset} & \textbf{Encoder} & N & Match\% & "
        r"$\lambdastar$ & Coverage & APSS & $U_\textsc{epi}$ AUROC \\"
    )
    lines.append(r"\midrule")

    for fi, gen_file in enumerate(EMBEDDING_TABLE_FILES):
        if fi > 0:
            lines.append(r"\midrule")
        if gen_file not in emb:
            continue
        per_encoder = emb[gen_file]
        dataset = _dataset_from_filename(gen_file)
        rows: list[str] = []
        # Header row uses multirow on the dataset name
        n_encoders_present = sum(1 for e in ENCODER_ORDER if e in per_encoder)
        lines.append(rf"\multirow{{{n_encoders_present}}}{{*}}{{{DATASET_PRETTY[dataset]}}}")
        for encoder in ENCODER_ORDER:
            if encoder not in per_encoder:
                continue
            m = per_encoder[encoder]
            n = m["n_samples"]
            match_pct = 100.0 * m["match_rate"]
            lam = m["lambda_star"]
            cov = m["coverage"]
            apss = m["apss"]
            auroc_epi = m.get("auroc_epi", float("nan"))
            auroc_str = "n/a" if is_nan(auroc_epi) else f"{auroc_epi:.3f}"
            line = (
                f" & {ENCODER_PRETTY[encoder]:<22} & {n} & "
                f"{match_pct:.1f} & {fmt_lambda(lam):>5} & "
                f"{cov:.3f} & {apss:.2f} & {auroc_str} \\\\"
            )
            lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def main() -> None:
    TABLES_OUT.mkdir(parents=True, exist_ok=True)

    stage3 = load_stage3()
    r_proto = load_r_protocol()
    emb = load_embedding()

    files = {
        "main_tabular.tex": build_table_main(stage3),
        "auroc_tabular.tex": build_table_auroc(stage3),
        "r_protocol_tabular.tex": build_table_r_protocol(r_proto),
        "embedding_tabular.tex": build_table_embedding(emb),
    }

    auto_header = (
        "% AUTO-GENERATED by src/make_paper_tables.py from JSON results.\n"
        "% Do not edit by hand - re-run the script after updating results.\n"
    )
    for name, content in files.items():
        out_path = TABLES_OUT / name
        out_path.write_text(auto_header + content, encoding="utf-8")
        print(f"wrote {out_path}  ({content.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
