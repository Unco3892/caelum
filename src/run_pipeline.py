#!/usr/bin/env python3
"""
CAELUM: End-to-end experiment pipeline.

Runs the full pipeline:
  1. Load data (TriviaQA / AmbigQA)
  2. Generate clarifications + answers (two-stage sampling)
  3. Compute ICE uncertainty decomposition
  4. Split into Val1 / Val2 / Test
  5. Calibrate (CAELUM + baselines)
  6. Evaluate and produce results

Usage:
    python run_pipeline.py --dataset triviaqa --n_samples 200
    python run_pipeline.py --dataset ambigqa  --n_samples 200
    python run_pipeline.py --dataset both     --n_samples 200
    python run_pipeline.py --dataset all      --n_samples 200
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# Suppress noisy third-party loggers & progress bars
import warnings as _warnings
_warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence tokenizers fork warning
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # disable HF download bars
os.environ["TQDM_DISABLE"] = "1"  # disable tqdm progress bars globally


# ---------------------------------------------------------------------------
# Tee: write to both stdout and log file, flushing every line
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8", buffering=1)  # line-buffered

    def write(self, msg):
        try:
            self.terminal.write(msg)
        except UnicodeEncodeError:
            self.terminal.write(msg.encode("ascii", errors="replace").decode("ascii"))
        self.terminal.flush()
        self.log.write(msg)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def fileno(self):
        # vLLM's engine startup (distributed process-group init) suppresses
        # NCCL setup noise via a context manager that redirects the real OS
        # file descriptor, which requires sys.stdout.fileno() to exist.
        # Delegate to the original terminal stream captured above.
        return self.terminal.fileno()

    def isatty(self):
        return self.terminal.isatty()


def _setup_logging(log_path: str):
    """Redirect stdout/stderr through Tee and configure logging."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout = Tee(log_path)
    sys.stderr = sys.stdout  # also capture stderr

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    for _noisy in ("httpx", "httpcore", "sentence_transformers", "transformers",
                   "huggingface_hub", "filelock", "urllib3", "fsspec", "safetensors",
                   "accelerate", "torch", "tqdm"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)
    return logging.getLogger("clear_llm")

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from caelum.paths import GENERATIONS_DIR, PLOTS_DIR  # noqa: E402

# Will be initialised in main() after parsing args
logger = logging.getLogger("clear_llm")

from caelum.utils import (
    SentenceEncoder, ensure_dir, load_ambigqa, load_ambiginst,
    load_condambigqa, load_json, load_opennq, load_popqa, load_simpleqa,
    load_triviaqa, save_json, split_data,
)
from caelum.generation import APIClarifier, HFModel, TRTModel, run_generation_pipeline
from caelum.uncertainty import (
    NLIEntailment, compute_uncertainty_for_dataset,
)
from caelum.calibration import CAELUMCalibrator
from caelum.evaluation import (
    evaluate, evaluate_diagnostic_ratio, diagnostic_ratio_summary,
    format_results_table,
    plot_lambda_curve, plot_uncertainty_scatter, plot_diagnostic_scatter,
)

# ---------------------------------------------------------------------------
# Paths (relative to workspace root)
# ---------------------------------------------------------------------------

WORKSPACE = SCRIPT_DIR.parent
SPECTRAL_DATA = WORKSPACE / "data" / "eval"

DATA_DIR = WORKSPACE / "data" / "eval"

DEFAULT_TRIVIAQA_PATH = str(DATA_DIR / "TriviaQA.json")
DEFAULT_AMBIGQA_PATH = str(DATA_DIR / "AmbigQA.json")
DEFAULT_OPENNQ_PATH = str(DATA_DIR / "OpenNQ.json")
DEFAULT_AMBIGINST_PATH = str(SPECTRAL_DATA / "AmbigInst.json")
DEFAULT_SIMPLEQA_PATH = str(DATA_DIR / "SimpleQA.json")
DEFAULT_POPQA_PATH = str(DATA_DIR / "PopQA.json")
DEFAULT_CONDAMBIGQA_PATH = str(DATA_DIR / "CondAmbigQA.json")

# Caches, generation logs and run summaries go to results/generations/.
OUTPUT_DIR = GENERATIONS_DIR


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_generate(
    args: argparse.Namespace,
    data: list,
    dataset_type: str,
    output_path: str,
) -> list:
    """Stage 1+2: Generate clarifications and answers.

    When ``args.clarif_provider`` is set to a provider name (gemini, groq),
    Stage 1 clarifications are generated via the remote API and only Stage 2
    answers use the local model.
    """
    if os.path.exists(output_path) and not args.regenerate:
        logger.info("Loading cached generations from %s", output_path)
        return load_json(output_path)

    logger.info("=== Stage: Generation (%s, %d samples) ===", dataset_type, len(data))
    t0 = time.time()

    # --- Optional API clarifier ---
    api_clarifier = None
    clarif_provider = getattr(args, "clarif_provider", "none")
    if clarif_provider and clarif_provider != "none":
        clarif_model = getattr(args, "clarif_model", None)
        api_clarifier = APIClarifier(
            provider=clarif_provider,
            model=clarif_model,
            env_path=WORKSPACE / ".env",
        )
        logger.info(
            "Using API clarifier: %s / %s", clarif_provider, api_clarifier.model
        )

    # --- Local model (always needed for answers, also for clarifications if no API) ---
    backend = getattr(args, "backend", "hf")
    if backend == "trt":
        engine_dir = getattr(args, "engine_dir", None)
        if not engine_dir:
            raise ValueError("--engine_dir is required when using --backend trt")
        model = TRTModel(
            engine_dir=engine_dir,
            tokenizer_name=args.model_name,
            max_new_tokens=256,
        )
    elif backend == "vllm":
        from caelum.generation import VLLMModel
        model = VLLMModel(
            model_name=args.model_name,
            max_new_tokens=256,
        )
    else:
        model = HFModel(
            model_name=args.model_name,
            device="cuda",
            max_new_tokens=256,
        )

    results = run_generation_pipeline(
        model=model,
        data=data,
        dataset_type=dataset_type,
        n_clarifications=args.n_clarifications,
        n_answers_per_clarification=args.n_answers,
        clarification_temperature=0.7,
        answer_temperature=0.5,
        api_clarifier=api_clarifier,
    )

    model.unload()
    save_json(results, output_path)
    elapsed = time.time() - t0
    logger.info("Saved generations to %s (%.1f min)", output_path, elapsed / 60)
    return results


def stage_uncertainty(
    args: argparse.Namespace,
    gen_results: list,
    output_path: str,
) -> list:
    """Stage 3+4: Compute uncertainty decomposition + correctness."""
    if os.path.exists(output_path) and not args.regenerate:
        logger.info("Loading cached uncertainties from %s", output_path)
        return load_json(output_path)

    logger.info("=== Stage: Uncertainty Decomposition (%d samples) ===", len(gen_results))
    t0 = time.time()
    encoder = SentenceEncoder(model_name="all-mpnet-base-v2")

    nli_model = None
    if args.use_nli:
        nli_model = NLIEntailment(model_name="cross-encoder/nli-deberta-v3-base")

    results = compute_uncertainty_for_dataset(
        gen_results,
        encoder=encoder,
        cluster_threshold=args.cluster_threshold,
        use_nli=args.use_nli,
        nli_model=nli_model,
        uncertainty_method=getattr(args, "uncertainty_method", "ce"),
        vne_gamma=getattr(args, "vne_gamma", 1.0),
    )

    if nli_model is not None:
        nli_model.unload()

    save_json(results, output_path)
    elapsed = time.time() - t0
    logger.info("Saved uncertainties to %s (%.1f min)", output_path, elapsed / 60)
    return results


def stage_calibrate_and_evaluate(
    args: argparse.Namespace,
    data: list,
    dataset_name: str,
) -> dict:
    """Stage 5+6: Split, calibrate, and evaluate."""
    logger.info("=== Stage: Calibrate & Evaluate (%s, %d samples) ===",
                dataset_name, len(data))

    val1, val2, test = split_data(data, val1_frac=0.4, val2_frac=0.3, seed=args.seed)
    logger.info("Split: Val1=%d, Val2=%d, Test=%d", len(val1), len(val2), len(test))

    weight_fn = getattr(args, "weight_fn", "additive")
    randomize = getattr(args, "randomize", False)
    all_results = {}

    # --- CAELUM (full optimisation) ---
    cal_clear = CAELUMCalibrator(
        alpha=args.alpha, seed=args.seed, weight_fn=weight_fn,
        randomize=randomize,
    )
    cal_clear.calibrate(val1, val2, verbose=True)
    res_clear = evaluate(test, cal_clear, dataset_name=f"CAELUM / {dataset_name}")
    all_results[f"CAELUM_{dataset_name}"] = res_clear

    # --- Stratified coverage diagnostic (CLEAR/JUCAL critique defense) ---
    strat = cal_clear.stratified_coverage(test, n_bins=4, by="U_total")
    logger.info("Stratified coverage on TEST (by U_total quartile):")
    for b in range(len(strat["coverage"])):
        logger.info(
            "  Q%d: n=%3d  coverage=%.3f  APSS=%.2f  edge<=%.3f",
            b + 1, strat["n"][b], strat["coverage"][b], strat["apss"][b],
            strat["bin_edges"][b + 1],
        )
    all_results[f"CAELUM_{dataset_name}"]["stratified_coverage"] = {
        "by": "U_total",
        "n_bins": 4,
        "coverage": list(map(float, strat["coverage"])),
        "apss": list(map(float, strat["apss"])),
        "n": list(map(int, strat["n"])),
        "edges": list(map(float, strat["bin_edges"])),
    }

    # --- R(x) diagnostic evaluation ---
    diag_results = evaluate_diagnostic_ratio(test, cal_clear)
    all_results[f"CAELUM_{dataset_name}"].update(diag_results)
    diag_summary = diagnostic_ratio_summary(test, cal_clear)
    logger.info("\n%s", diag_summary)

    # Plot lambda curve
    ensure_dir(str(PLOTS_DIR))
    plot_lambda_curve(
        cal_clear,
        save_path=str(PLOTS_DIR / f"lambda_curve_{dataset_name}.png"),
        title=f"Lambda Optimisation — {dataset_name}",
    )

    # Plot R(x) diagnostic scatter
    plot_diagnostic_scatter(
        test, cal_clear,
        save_path=str(PLOTS_DIR / f"diagnostic_scatter_{dataset_name}.png"),
        title=f"R(x) Diagnostic — {dataset_name}",
    )

    # --- Vanilla conformal (lambda=0) ---
    cal_vanilla = CAELUMCalibrator.vanilla_conformal(
        alpha=args.alpha, weight_fn=weight_fn, randomize=randomize,
    )
    cal_vanilla.calibrate(val1, val2, verbose=True)
    res_vanilla = evaluate(test, cal_vanilla, dataset_name=f"Vanilla / {dataset_name}")
    all_results[f"Vanilla_{dataset_name}"] = res_vanilla

    # --- Epistemic-only (lambda -> inf) ---
    cal_epi = CAELUMCalibrator.epistemic_only(
        alpha=args.alpha, weight_fn=weight_fn, randomize=randomize,
    )
    cal_epi.calibrate(val1, val2, verbose=True)
    res_epi = evaluate(test, cal_epi, dataset_name=f"Epistemic-only / {dataset_name}")
    all_results[f"EpistemicOnly_{dataset_name}"] = res_epi

    # --- Semantic Entropy baseline (no prediction sets, just AUROC) ---
    from caelum.evaluation import (
        auroc_hallucination, auroc_ambiguity,
        auroc_total_hallucination, auroc_total_ambiguity,
    )
    all_results[f"SemanticEntropy_{dataset_name}"] = {
        "coverage": float("nan"),
        "apss": float("nan"),
        "auroc_epi_hallucination": auroc_total_hallucination(test),
        "auroc_ale_ambiguity": auroc_total_ambiguity(test),
        "optimal_lambda": float("nan"),
        "tau_hat": float("nan"),
    }

    # Uncertainty scatter
    color_by = "ambiguity" if "ambig" in dataset_name.lower() else "correctness"
    plot_uncertainty_scatter(
        test,
        color_by=color_by,
        save_path=str(PLOTS_DIR / f"scatter_{dataset_name}.png"),
        title=f"Uncertainty Decomposition — {dataset_name}",
    )

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dataset(args: argparse.Namespace, dataset_type: str) -> dict:
    """Run the full pipeline for one dataset."""
    t0 = time.time()
    dataset_name = dataset_type.capitalize()

    # 1. Load data
    loaders = {
        "triviaqa": (load_triviaqa, lambda a: a.triviaqa_path or DEFAULT_TRIVIAQA_PATH),
        "ambigqa": (load_ambigqa, lambda a: a.ambigqa_path or DEFAULT_AMBIGQA_PATH),
        "opennq": (load_opennq, lambda a: a.opennq_path or DEFAULT_OPENNQ_PATH),
        "ambiginst": (load_ambiginst, lambda a: a.ambiginst_path or DEFAULT_AMBIGINST_PATH),
        "simpleqa": (load_simpleqa, lambda a: a.simpleqa_path or DEFAULT_SIMPLEQA_PATH),
        "popqa": (load_popqa, lambda a: a.popqa_path or DEFAULT_POPQA_PATH),
        "condambigqa": (load_condambigqa, lambda a: a.condambigqa_path or DEFAULT_CONDAMBIGQA_PATH),
    }
    # Ambiguous datasets use ambigqa prompts; factual use triviaqa prompts
    prompt_type_map = {
        "triviaqa": "triviaqa", "ambigqa": "ambigqa", "opennq": "triviaqa",
        "ambiginst": "ambigqa", "simpleqa": "triviaqa", "popqa": "triviaqa",
        "condambigqa": "ambigqa",
    }
    loader_fn, path_fn = loaders[dataset_type]
    data_path = path_fn(args)
    data = loader_fn(data_path, n_samples=args.n_samples, seed=args.seed)
    dataset_type_for_prompts = prompt_type_map[dataset_type]

    ensure_dir(str(OUTPUT_DIR))
    model_tag = args.model_name.split("/")[-1].lower().replace("-", "_")
    unc_method = getattr(args, "uncertainty_method", "ce")
    clarif_prov = getattr(args, "clarif_provider", "none")
    if clarif_prov and clarif_prov != "none":
        # If an explicit model override was given, encode it in the suffix so
        # different API models produce separate cache files.
        # e.g.  gemini-2.5-flash-lite → _gemini          (default, backward compat)
        #       gemini-2.5-pro        → _gemini_pro
        #       gemini-2.5-flash      → _gemini_flash
        _cmodel = getattr(args, "clarif_model", None) or ""
        _provider_cfg = {
            "gemini": "gemini-2.5-flash-lite",
            "groq": "llama-3.1-8b-instant",
        }
        _default_model = _provider_cfg.get(clarif_prov, "")
        if _cmodel and _cmodel != _default_model:
            # Derive a short tag from the model name, e.g.
            # "gemini-2.5-pro" → "pro", "gemini-2.5-flash" → "flash"
            _normalized = _cmodel.lower().replace("-", "_").replace(".", "_")
            _model_parts = _normalized.split("_")
            # Drop parts that match the provider name or version numbers for brevity
            _tag_parts = [p for p in _model_parts if p not in ("gemini", "groq", "2", "5")]
            _model_tag_short = "_".join(_tag_parts) if _tag_parts else _normalized
            clarif_suffix = f"_{clarif_prov}_{_model_tag_short}"
        else:
            clarif_suffix = f"_{clarif_prov}"
    else:
        clarif_suffix = ""
    gen_path = str(OUTPUT_DIR / f"gen_{dataset_type}_{model_tag}{clarif_suffix}_{args.n_samples}.json")
    unc_path = str(OUTPUT_DIR / f"unc_{dataset_type}_{model_tag}_{unc_method}{clarif_suffix}_{args.n_samples}.json")

    # 2. Generate
    gen_results = stage_generate(args, data, dataset_type_for_prompts, gen_path)

    # 3. Uncertainty decomposition
    unc_results = stage_uncertainty(args, gen_results, unc_path)

    # 4. Calibrate & evaluate
    all_results = stage_calibrate_and_evaluate(args, unc_results, dataset_name)

    elapsed = time.time() - t0
    logger.info("Finished %s in %.1f minutes.", dataset_name, elapsed / 60)
    return all_results


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and all GPUs) before sampling.

    This fixes Hugging Face sampling for a given seed, GPU type and software
    stack. vLLM and TensorRT-LLM samplers keep their own state.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    global logger

    parser = argparse.ArgumentParser(description="CAELUM Experiment Pipeline")

    # Dataset
    parser.add_argument("--dataset", type=str, default="both",
                        choices=["triviaqa", "ambigqa", "opennq", "ambiginst",
                                 "simpleqa", "popqa", "condambigqa",
                                 "both", "all"],
                        help="Which dataset(s) to run. 'both'=triviaqa+ambigqa, "
                             "'all'=all datasets.")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of questions per dataset.")
    parser.add_argument("--triviaqa_path", type=str, default=None)
    parser.add_argument("--ambigqa_path", type=str, default=None)
    parser.add_argument("--opennq_path", type=str, default=None)
    parser.add_argument("--ambiginst_path", type=str, default=None)
    parser.add_argument("--simpleqa_path", type=str, default=None)
    parser.add_argument("--popqa_path", type=str, default=None)
    parser.add_argument("--condambigqa_path", type=str, default=None)

    # Model / backend
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen3-8B",
                        help="HuggingFace model name for generation.")
    parser.add_argument("--backend", type=str, default="hf",
                        choices=["hf", "trt", "vllm"],
                        help="Inference backend: 'hf' (HuggingFace), 'trt' "
                             "(TensorRT-LLM), or 'vllm' (cluster large-model "
                             "rows; requires the dedicated vllm env).")
    parser.add_argument("--engine_dir", type=str, default=None,
                        help="TensorRT engine directory (required for --backend trt).")

    # API clarification (Stage 1 only — answers always use local model)
    parser.add_argument("--clarif_provider", type=str, default="none",
                        choices=["none", "gemini", "groq"],
                        help="API provider for Stage 1 clarifications. "
                             "'none' uses local model for everything.")
    parser.add_argument("--clarif_model", type=str, default=None,
                        help="Model name for the API clarifier. "
                             "Defaults: gemini-2.5-flash-lite (gemini), "
                             "llama-3.1-8b-instant (groq).")

    # Generation
    parser.add_argument("--n_clarifications", type=int, default=5)
    parser.add_argument("--n_answers", type=int, default=10)

    # Uncertainty decomposition
    parser.add_argument("--uncertainty_method", type=str, default="vne",
                        choices=["ce", "vne"],
                        help="Uncertainty method: 'vne' (Von Neumann entropy, default) or 'ce' (cluster entropy, ablation only).")
    parser.add_argument("--vne_gamma", type=float, default=1.0,
                        help="RBF kernel bandwidth for VNE (default 1.0).")

    # Clustering
    parser.add_argument("--cluster_threshold", type=float, default=0.7,
                        help="Cosine similarity threshold for clustering.")
    parser.add_argument("--use_nli", action="store_true",
                        help="Use NLI-based clustering instead of embeddings.")

    # Calibration
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Significance level (1-alpha = target coverage).")
    parser.add_argument("--weight_fn", type=str, default="additive",
                        choices=["additive", "multiplicative", "inverse"],
                        help="Weight function: 'additive' (default, paper), "
                             "'multiplicative', or 'inverse' (ablation only).")
    parser.add_argument("--randomize", action="store_true",
                        help="Enable randomized APS scoring. Default: deterministic "
                             "(matches paper math; randomization is an ablation).")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for question sampling, data splits and Hugging Face "
                             "generation sampling.")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force regeneration even if cached results exist.")

    parser.add_argument("--write-markdown", action="store_true",
                        help="Also write Markdown reports (off by default).")
    args = parser.parse_args()

    # --- Set up Tee logging ---
    model_tag = args.model_name.split("/")[-1].lower().replace("-", "_")
    log_filename = f"pipeline_{model_tag}_{args.uncertainty_method}_{args.dataset}_{args.n_samples}.txt"
    log_path = str(OUTPUT_DIR / log_filename)
    ensure_dir(str(OUTPUT_DIR))
    logger = _setup_logging(log_path)

    t_total_start = time.time()

    logger.info("=" * 70)
    logger.info("CAELUM Experiment Pipeline")
    logger.info("=" * 70)
    logger.info("Log file: %s", log_path)
    logger.info("Config: %s", json.dumps(vars(args), indent=2))
    seed_everything(args.seed)
    logger.info("Seeded Python, NumPy and torch with %d before generation.", args.seed)

    combined_results = {}

    all_datasets = ["triviaqa", "ambigqa", "opennq", "ambiginst",
                     "simpleqa", "popqa", "condambigqa"]
    both_datasets = ["triviaqa", "ambigqa"]

    if args.dataset == "all":
        datasets_to_run = all_datasets
    elif args.dataset == "both":
        datasets_to_run = both_datasets
    else:
        datasets_to_run = [args.dataset]

    for ds in datasets_to_run:
        res = run_dataset(args, ds)
        combined_results.update(res)

    # Summary table
    table = format_results_table(combined_results)
    logger.info("\n%s", table)

    # Save results
    results_path = str(OUTPUT_DIR / log_filename.replace(".txt", "_summary.json"))
    save_json(combined_results, results_path)
    logger.info("Saved results to %s", results_path)

    if args.write_markdown:
        table_path = str(OUTPUT_DIR / log_filename.replace(".txt", "_table.md"))
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(table)
        logger.info("Saved results table to %s", table_path)

    total_elapsed = time.time() - t_total_start
    logger.info("=" * 70)
    logger.info("TOTAL RUNTIME: %.1f minutes (%.0f seconds)", total_elapsed / 60, total_elapsed)
    logger.info("Log saved to: %s", log_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
