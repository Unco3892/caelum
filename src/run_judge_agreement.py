#!/usr/bin/env python3
"""Agreement between two LLM judges on the heuristic matcher's negatives.

The heuristic matcher marks an item unmatched (``correct_cluster_idx = -1``)
when no sampled cluster matches the reference. In ``full`` mode,
``src/run_llm_judge_rematch.py`` sends every such item to an LLM judge, which
may assign it to a cluster (a flip); heuristic positives are never re-judged.
Two judges were run on the same 30 cells (five generators x six datasets):

    * first judge, gemini-2.5-flash-lite: ``results/llm_judged_unc_full/``,
      the labels used by the judged analyses;
    * second, independent judge, gemini-2.5-flash:
      ``results/llm_judged_unc_flash/``.

For each cell the script reads the heuristic cache (``results/generations/``)
and both judged caches, checks that they are aligned question by question,
that heuristic positives are unchanged and that every heuristic negative has a
judge audit record, and counts

    * heuristic negatives (one judge call each) and each judge's flips;
    * agreement: both judges give the same support verdict (matched or
      unmatched) on a heuristic negative;
    * the match-rate shift, first-judge minus second-judge match rate, in
      percentage points.

Per-cell counts are cross-checked against the run summaries
``results/llm_judge_match_rates_*.json``. The appendix on reference-matching
sensitivity reports the totals: 4,118 negatives, 168 first-judge flips and
97.4% agreement, with cell shifts of at most 2.2 percentage points.

Writes ``results/judge_agreement.json``. Run from the repository root::

    python src/run_judge_agreement.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, JUDGED_DIR, RESULTS_DIR, repo_relative  # noqa: E402

SECOND_JUDGE_DIR = RESULTS_DIR / "llm_judged_unc_flash"
MODELS = ["qwen3_8b", "ministral_3_8b_instruct_2512", "gemma_4_e4b_it", "qwen3_14b", "qwen3_32b"]
DATASETS = {"triviaqa": 300, "ambigqa": 200, "opennq": 300,
            "ambiginst": 304, "simpleqa": 500, "popqa": 500}

DEFINITIONS = {
    "negative_cases": "items the heuristic matcher left unmatched (correct_cluster_idx = -1); each received one call from each judge",
    "flips": "heuristic negatives that a judge assigned to a sampled cluster",
    "agreement": "heuristic negatives on which both judges give the same support verdict (matched or unmatched)",
    "same_cluster_agreement": "heuristic negatives on which both judges give the same correct_cluster_idx; stricter than agreement",
    "gain_pp": "judged minus heuristic match rate of a cell, in percentage points",
    "shift_pp": "first-judge minus second-judge match rate of a cell, in percentage points",
}


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pp(count: int, n: int) -> float:
    return count / n * 100


def judge_model(records: list[dict], path: Path) -> str | None:
    models = {r["llm_judge_audit"]["model"] for r in records if r.get("llm_judge_audit")}
    assert len(models) <= 1, (path, models)
    return models.pop() if models else None


def compare_cell(heuristic: list[dict], first: list[dict], second: list[dict]) -> dict:
    """Count negatives, flips and verdict agreement for one aligned cell."""
    assert len(heuristic) == len(first) == len(second)
    counts = dict(negative_cases=0, first_judge_flips=0, second_judge_flips=0,
                  both_judges_flipped=0, agreement_count=0, same_cluster_agreement_count=0)
    for h, a, b in zip(heuristic, first, second):
        assert h["question"] == a["question"] == b["question"]
        if h["correct_cluster_idx"] >= 0:
            assert a["correct_cluster_idx"] == b["correct_cluster_idx"] == h["correct_cluster_idx"]
            assert not a.get("llm_judge_audit") and not b.get("llm_judge_audit")
            continue
        for record in (a, b):
            audit = record["llm_judge_audit"]
            assert audit["called"] and audit["error"] is None, audit
            assert audit["pre_correct_cluster_idx"] == -1
            assert audit["post_correct_cluster_idx"] == record["correct_cluster_idx"]
        matched_a = a["correct_cluster_idx"] >= 0
        matched_b = b["correct_cluster_idx"] >= 0
        counts["negative_cases"] += 1
        counts["first_judge_flips"] += matched_a
        counts["second_judge_flips"] += matched_b
        counts["both_judges_flipped"] += matched_a and matched_b
        counts["agreement_count"] += matched_a == matched_b
        counts["same_cluster_agreement_count"] += a["correct_cluster_idx"] == b["correct_cluster_idx"]
    return counts


def load_summaries(rates_dir: Path) -> dict[str, dict[str, tuple[dict, Path]]]:
    """Per-cell rows of the judge run summaries, keyed by judge model."""
    by_model: dict[str, dict[str, tuple[dict, Path]]] = {}
    for path in sorted(rates_dir.glob("llm_judge_match_rates_*.json")):
        summary = load(path)
        cells = by_model.setdefault(summary["model"], {})
        for key, row in summary["cells"].items():
            assert key not in cells, f"{key} appears in two {summary['model']} summaries"
            cells[key] = (row, path)
    return by_model


def check_summary(row: dict, n: int, negatives: int, flips: int,
                  heuristic_rate: float, judged_rate: float) -> None:
    assert row["mode"] == "full" and row["n_samples"] == n
    assert row["n_unmatched"] == row["n_targets"] == row["n_calls"] == negatives
    assert row["n_flipped"] == flips and row["n_skipped_budget"] == 0 and not row["errors"]
    assert math.isclose(row["original_match_rate"], heuristic_rate, abs_tol=1e-12)
    assert math.isclose(row["judged_match_rate"], judged_rate, abs_tol=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations_dir", type=Path, default=GENERATIONS_DIR,
                        help="Directory with the heuristic unc_*_vne_*.json caches.")
    parser.add_argument("--judged_dir", type=Path, default=JUDGED_DIR,
                        help="Directory with the first judge's caches.")
    parser.add_argument("--second_judge_dir", type=Path, default=SECOND_JUDGE_DIR,
                        help="Directory with the second judge's caches.")
    parser.add_argument("--rates_dir", type=Path, default=RESULTS_DIR,
                        help="Directory with the llm_judge_match_rates_*.json run summaries.")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "judge_agreement.json",
                        help="Output JSON path.")
    args = parser.parse_args()

    summaries = load_summaries(args.rates_dir)
    first_model = second_model = None
    rows = []
    for model in MODELS:
        for dataset, n in DATASETS.items():
            name = f"unc_{dataset}_{model}_vne_{n}.json"
            heuristic = load(args.generations_dir / name)
            first = load(args.judged_dir / name)
            second = load(args.second_judge_dir / name)
            for found, cache in [(judge_model(first, name), "first"),
                                 (judge_model(second, name), "second")]:
                if found is None:
                    continue
                if cache == "first":
                    first_model = first_model or found
                    assert found == first_model, (name, found)
                else:
                    second_model = second_model or found
                    assert found == second_model, (name, found)
            counts = compare_cell(heuristic, first, second)
            matched = sum(h["correct_cluster_idx"] >= 0 for h in heuristic)
            assert matched + counts["negative_cases"] == n
            rates = {
                "heuristic_match_rate": matched / n,
                "first_judge_match_rate": (matched + counts["first_judge_flips"]) / n,
                "second_judge_match_rate": (matched + counts["second_judge_flips"]) / n,
            }
            rows.append({"cell_key": f"{model}__{dataset}", "model": model,
                         "dataset": dataset, "cache": name, "n": n, **counts, **rates,
                         "first_judge_gain_pp": pp(counts["first_judge_flips"], n),
                         "second_judge_gain_pp": pp(counts["second_judge_flips"], n),
                         "shift_pp": pp(counts["first_judge_flips"] - counts["second_judge_flips"], n)})

    used_summaries = set()
    for row in rows:
        for judge, model in [("first", first_model), ("second", second_model)]:
            summary, path = summaries[model][row["cell_key"]]
            check_summary(summary, row["n"], row["negative_cases"], row[f"{judge}_judge_flips"],
                          row["heuristic_match_rate"], row[f"{judge}_judge_match_rate"])
            row[f"{judge}_judge_summary"] = repo_relative(path)
            used_summaries.add(repo_relative(path))

    total = {key: sum(row[key] for row in rows) for key in
             ["negative_cases", "first_judge_flips", "second_judge_flips", "both_judges_flipped",
              "agreement_count", "same_cluster_agreement_count"]}
    max_shift = max(rows, key=lambda row: abs(row["shift_pp"]))
    max_gain = max(rows, key=lambda row: row["first_judge_gain_pp"])
    payload = {
        "scope": (f"{len(rows)} cells ({len(MODELS)} generators x {len(DATASETS)} datasets); "
                  "only heuristic-negative items are compared. Heuristic positives are not "
                  "re-judged, so this checks false negatives of the matcher, not its false positives."),
        "definitions": DEFINITIONS,
        "inputs": {
            "generations_dir": repo_relative(args.generations_dir),
            "first_judge_dir": repo_relative(args.judged_dir),
            "second_judge_dir": repo_relative(args.second_judge_dir),
            "run_summaries": sorted(used_summaries),
        },
        "first_judge_model": first_model,
        "second_judge_model": second_model,
        "n_cells": len(rows),
        **total,
        "agreement_rate": total["agreement_count"] / total["negative_cases"],
        "same_cluster_agreement_rate": (total["same_cluster_agreement_count"]
                                        / total["negative_cases"]),
        "max_cell_shift_pp": abs(max_shift["shift_pp"]),
        "max_cell_shift_cell": max_shift["cell_key"],
        "max_first_judge_gain_pp": max_gain["first_judge_gain_pp"],
        "max_first_judge_gain_cell": max_gain["cell_key"],
        "cells": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{payload['n_cells']} cells: {total['negative_cases']} heuristic negatives; "
          f"flips {total['first_judge_flips']} ({first_model}) and "
          f"{total['second_judge_flips']} ({second_model})")
    print(f"agreement {total['agreement_count']}/{total['negative_cases']} = "
          f"{payload['agreement_rate']:.1%}; max cell shift "
          f"{payload['max_cell_shift_pp']:.1f} pp ({payload['max_cell_shift_cell']})")
    print(f"Wrote {repo_relative(args.output)}")


if __name__ == "__main__":
    main()
