#!/usr/bin/env python3
"""Qwen3-8B versus Qwen3-32B heuristic matchability on the fresh FP16 caches.

Recomputes the generator-scale support table (``tab:fresh_scale_support``,
``paper/neurips/tables/scale_support_fresh.tex``). The inputs are the twelve
uncertainty caches (``unc_*``) of the Qwen3-8B and Qwen3-32B cells of run
caelum_fresh_20261103_0633: generation seed 42, FP16 without 4-bit
quantisation, five clarifications and ten answers per clarification, VNE with
the heuristic reference matcher. They are stored in
``results/generations/fresh_scale_20261103/``.

A question counts as matched when its heuristic ``correct_cluster_idx`` is
>= 0, i.e. some generated cluster matches a reference. This is all-query
matchability, not conformal coverage. The pooled row weights all 2,104
questions equally. The script checks each cache's record count and sha256
(against the run's per-cell SUCCESS.json records) and that both generators see
the same ordered questions and references.

Writes ``results/fresh_scale_support.json``. Run from the repository root::

    python src/run_fresh_scale_support.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, RESULTS_DIR, content_sha256, repo_relative  # noqa: E402

RUN_ID = "caelum_fresh_20261103_0633"
GENERATION_SEED = 42
DEFAULT_INPUT_DIR = GENERATIONS_DIR / "fresh_scale_20261103"
DEFAULT_OUTPUT = RESULTS_DIR / "fresh_scale_support.json"

MODELS = [("qwen3_8b", "Qwen3-8B"), ("qwen3_32b", "Qwen3-32B")]
DATASETS = [  # (cache key, display name, number of questions), in table order
    ("triviaqa", "TriviaQA", 300),
    ("ambigqa", "AmbigQA", 200),
    ("opennq", "OpenNQ", 300),
    ("ambiginst", "AmbigInst", 304),
    ("simpleqa", "SimpleQA", 500),
    ("popqa", "PopQA", 500),
]

# sha256 of each cache as recorded in cells/<cell>/SUCCESS.json ("outputs") and
# analysis/input_inventory.json of run caelum_fresh_20261103_0633.
EXPECTED = {
    "unc_triviaqa_qwen3_8b_vne_300.json": ("00_qwen3_8b_triviaqa", "ff8ec4357e178fd63676905ca49e1da7fc2f67cac0e88927899af406d832d928"),
    "unc_ambigqa_qwen3_8b_vne_200.json": ("01_qwen3_8b_ambigqa", "7e512cbf2b20a4c5b3471254fef9debdf0a502763c6d827a09f6a1bc0975fb37"),
    "unc_opennq_qwen3_8b_vne_300.json": ("02_qwen3_8b_opennq", "a97b8d68e3868e2dc8f02fa79d7ff4b57a9e601aa2789bcf5bb78ea250bb0d8f"),
    "unc_ambiginst_qwen3_8b_vne_304.json": ("03_qwen3_8b_ambiginst", "515d7889c7920870bf4dfc5e8acc8ca984f03f92176d498aa08ea78e59d919bb"),
    "unc_simpleqa_qwen3_8b_vne_500.json": ("04_qwen3_8b_simpleqa", "b43cb0bcd3062e109e231fd1ca620618213e5669eb7f4ad92be10e9bc7af1f86"),
    "unc_popqa_qwen3_8b_vne_500.json": ("05_qwen3_8b_popqa", "25f9d16e029db7bd692b454445146f25be56b44189a89cff395fedc31db7807c"),
    "unc_triviaqa_qwen3_32b_vne_300.json": ("24_qwen3_32b_triviaqa", "45533e84c22631e3ab2ab5ec1d2c952e06ce0559ae60f45dbf5b6fab20a01d24"),
    "unc_ambigqa_qwen3_32b_vne_200.json": ("25_qwen3_32b_ambigqa", "9b61942c6d4acdf0984106d3ff328adb3885b1d24ed60df39ccb31525e3ee501"),
    "unc_opennq_qwen3_32b_vne_300.json": ("26_qwen3_32b_opennq", "0a537a2995513c4df01cf0577d5855f8ff899bd89eb7143ed2178d7e77ec068e"),
    "unc_ambiginst_qwen3_32b_vne_304.json": ("27_qwen3_32b_ambiginst", "aef7c9bed73e2a9009cf53be1f95bc0410f56cdb0c76676bba08d34a6657f10f"),
    "unc_simpleqa_qwen3_32b_vne_500.json": ("28_qwen3_32b_simpleqa", "62cc7ccdd6b25e384264157dbf83613979b3208737150f6e95c7108843387beb"),
    "unc_popqa_qwen3_32b_vne_500.json": ("29_qwen3_32b_popqa", "9d46804b001295e003d5b5a209927a3506a2a651175f03e980e0ad783b7b2ef8"),
}


def load_cache(path: Path, n_expected: int) -> tuple[list[dict], dict]:
    """Load one cache, check its record count and record its sha256."""
    raw = path.read_bytes()
    items = json.loads(raw.decode("utf-8"))
    if len(items) != n_expected:
        raise SystemExit(f"{path}: {len(items)} records, expected {n_expected}.")
    digest = content_sha256(path)
    cell_id, expected = EXPECTED[path.name]
    if digest != expected:
        print(f"WARNING: {path.name} sha256 {digest} differs from the run's record "
              f"{expected}; these are not the {RUN_ID} caches.", file=sys.stderr)
    info = {"file": repo_relative(path), "cell_id": cell_id, "sha256": digest,
            "expected_sha256": expected, "hash_ok": digest == expected, "n_records": len(items)}
    return items, info


def alignment_key(item: dict) -> str:
    """Question and references that must agree across the two generators.

    AmbigQA and AmbigInst records carry no ``datapoint_id`` or ``gt_answer``.
    """
    return json.dumps([item["question_id"], item.get("datapoint_id"), item["question"],
                       item.get("gt_answer"), item["gt_answers"]], sort_keys=True)


def n_matched(items: list[dict]) -> int:
    """Questions whose reference matches some generated cluster (heuristic matcher)."""
    return sum(1 for item in items if int(item.get("correct_cluster_idx", -1)) >= 0)


def format_cell(matched: int, n: int, mark: str) -> str:
    rate = f"{100 * matched / n:.1f}\\%"
    return f"${matched}/{n}$ ({mark}{{{rate}}})" if mark else f"${matched}/{n}$ ({rate})"


def latex_row(name: str, n_label: str, counts: list[int], n: int) -> str:
    """One table row; bold marks the higher and underline the lower support rate."""
    small, large = counts
    marks = ["", ""]
    if small != large:
        hi = int(large > small)
        marks[hi], marks[1 - hi] = "\\textbf", "\\underline"
    delta = 100 * (large - small) / n
    cells = " & ".join(format_cell(c, n, m) for c, m in zip(counts, marks))
    return f"{name} & {n_label} & {cells} & ${delta:+.1f}$ \\\\"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations_dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Folder holding the twelve unc_* caches "
                             "(default: results/generations/fresh_scale_20261103).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSON (default: results/fresh_scale_support.json).")
    args = parser.parse_args()

    rows, inputs, latex = [], [], []
    totals = {tag: 0 for tag, _ in MODELS}
    n_total = 0
    for key, name, n in DATASETS:
        per_model, keys = {}, []
        for tag, label in MODELS:
            path = args.generations_dir / f"unc_{key}_{tag}_vne_{n}.json"
            items, info = load_cache(path, n)
            inputs.append(info)
            keys.append([alignment_key(item) for item in items])
            m = n_matched(items)
            totals[tag] += m
            per_model[tag] = {"model": label, "file": info["file"], "n_matched": m,
                              "rate": m / n, "rate_pct": round(100 * m / n, 1)}
        if keys[0] != keys[1]:
            raise SystemExit(f"{name}: the two caches differ in ordered questions or references.")
        counts = [per_model[tag]["n_matched"] for tag, _ in MODELS]
        delta_pp = 100 * (counts[1] - counts[0]) / n
        rows.append({"dataset": name, "n": n, "identical_ordered_questions_references": True,
                     **per_model, "delta_pp": delta_pp, "delta_pp_rounded": round(delta_pp, 1)})
        latex.append(latex_row(name, str(n), counts, n))
        n_total += n

    counts = [totals[tag] for tag, _ in MODELS]
    delta_pp = 100 * (counts[1] - counts[0]) / n_total
    pooled = {"dataset": "All questions", "n": n_total,
              **{tag: {"model": label, "n_matched": totals[tag], "rate": totals[tag] / n_total,
                       "rate_pct": round(100 * totals[tag] / n_total, 1)} for tag, label in MODELS},
              "delta_pp": delta_pp, "delta_pp_rounded": round(delta_pp, 1)}
    latex.append(latex_row("\\textbf{All questions}", f"{n_total:,}", counts, n_total))

    result = {
        "description": "All-query heuristic matchability (correct_cluster_idx >= 0) of Qwen3-8B "
                       "and Qwen3-32B on fresh FP16 caches; delta_pp = 32B rate minus 8B rate "
                       "in percentage points. Not conformal coverage.",
        "run_id": RUN_ID,
        "generation_seed": GENERATION_SEED,
        "generations_dir": repo_relative(args.generations_dir),
        "all_hashes_match_run_record": all(info["hash_ok"] for info in inputs),
        "datasets": rows,
        "pooled": pooled,
        "latex_rows": latex,
        "inputs": inputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"{'Dataset':<14}{'N':>6}  {'Qwen3-8B':<18}{'Qwen3-32B':<18}{'Delta (pp)':>10}")
    for row in rows + [pooled]:
        cells = [f"{row[tag]['n_matched']}/{row['n']} ({row[tag]['rate_pct']:.1f}%)"
                 for tag, _ in MODELS]
        print(f"{row['dataset']:<14}{row['n']:>6}  {cells[0]:<18}{cells[1]:<18}"
              f"{row['delta_pp']:>+10.1f}")
    print("\nLaTeX rows:")
    print("\n".join(latex))
    print(f"\nWrote {repo_relative(args.output)}")


if __name__ == "__main__":
    main()
