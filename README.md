# Matchability-Aware Conformal Prediction for Open-Ended Language Model Generation

**[Ilia Azizi](https://iliaazizi.com)**<sup>1,2</sup> &nbsp; <sup>1</sup> [University of Lausanne](https://www.unil.ch/unil/en/home.html) &nbsp; <sup>2</sup> [BegooAI](https://begoo.ai/)

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-8c1b13)](docs/paper.pdf)
[![Project page](https://img.shields.io/badge/Project-page-1f6feb)](https://unco3892.github.io/caelum/)
[![Poster](https://img.shields.io/badge/Poster-PDF-555)](docs/poster.pdf)
[![BibTeX](https://img.shields.io/badge/Cite-BibTeX-2ea44f)](#cite-this-work)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**TL;DR**

- CAELUM builds semantic prediction sets. It samples answers under several
  clarifications of a question, groups them by meaning, and uses uncertainty
  estimates to select groups, separating ambiguity in the question (aleatoric)
  from gaps in the model's knowledge (epistemic).
- *Matchability*, the share of questions whose reference answer matches a
  sampled cluster, caps coverage. CAELUM reaches that ceiling in 16 of the 18
  main model–dataset combinations.
- A classifier predicts matchability from the question before any answer is
  generated. Across 30 combinations, average coverage is 0.74 on the questions
  it predicts as matchable and 0.65 overall, against a nominal target of 0.90.

<details>
<summary><b>Abstract</b></summary>

Large language models (LLMs) often do not produce calibrated
uncertainty estimates for their answers. We propose CAELUM, a framework that
builds semantic prediction sets while separating aleatoric from epistemic
uncertainty. CAELUM samples answers using an ensemble of clarification prompts,
clusters them by semantic equivalence, and estimates the aleatoric and
epistemic components via kernel-based von Neumann entropy before calibrating
nonconformity scores with split conformal prediction. We use a weight
$w_\lambda(x)=1+U_{\mathrm{ale}}(x)+\lambda\,U_{\mathrm{epi}}(x)$ to control for
the contribution of the epistemic uncertainty of each instance and fit a
coefficient $\lambda^\star$ to minimize prediction set size. When the objective
is nonflat and selection is stable across splits, $\lambda^\star$ could be a
diagnostic, where large values may indicate epistemic dominance (retrieve,
abstain) and $\lambda^\star\to 0$ aleatoric dominance (clarify).
For individual queries, a log ratio $R(x)$ compares the weighted epistemic
component with the aleatoric component. Theoretically, we demonstrate that the
coverage is capped by *matchability*, the fraction of items whose reference
answer appears in any sampled cluster. Empirically, this matchability can be
predicted from the original question features using a lightweight classifier.
We use those predictions to define a stratum and derive a conformal coverage
bound within it. Across 30 LLM and dataset combinations (4–32B), mean coverage
is 0.74 in the learned stratum versus 0.65 overall, averaged over the 78
evaluations using the learned classifier.

</details>

## Installation

CAELUM runs on Python 3.11. Clone the repository and install the tested
dependency versions with Conda:

```bash
git clone https://github.com/Unco3892/caelum.git
cd caelum
conda create -n caelum python=3.11 -y
conda activate caelum
python -m pip install -r requirements.txt
```

The versions in `requirements.txt` come from the Linux environment used for the
generation runs (Python 3.11.15, NVIDIA A100). A fresh environment built
from this file passed `pip check`, ran every entry point's `--help`, completed a
short Qwen3-8B generation run and reproduced the cached analyses. Other
operating systems are untested, and the PyTorch CUDA build and `bitsandbytes`
are platform-specific: install the PyTorch build that matches your CUDA
version first if the default one does not. The cluster environment is pinned
separately in [`slurm/requirements-dcsr.lock.txt`](slurm/requirements-dcsr.lock.txt).

Re-downloading the source datasets also needs `python -m pip install datasets`.
The evaluation inputs are already in `data/eval/`, so this step is optional.

## Environment variables

**None are needed to recompute the reported analyses from the committed
caches.** Those analyses read `results/` and need only the pretrained sentence
encoders, which Hugging Face downloads on first use.

New generation, LLM judging and external clarifiers need credentials. Copy the
template and fill in only what you use:

```bash
cp .env.example .env
```

`.env` is ignored by Git. The API clients read it from the repository root and
fall back to the process environment.

| Variable | Needed for |
|---|---|
| `HF_TOKEN` | Downloading checkpoints that require accepting a license on Hugging Face. `huggingface_hub` reads it from the shell environment (or use `huggingface-cli login`); the Slurm launchers export it from `$WORK_ROOT/.env`. |
| `GEMINI_API_KEY` | The LLM judge (`src/llm_judge_matcher.py`, `src/run_llm_judge_rematch.py`) and `run_pipeline.py --clarif_provider gemini`. |
| `GROQ_API_KEY`, `OPENAI_API_KEY` | `run_pipeline.py --clarif_provider groq` or `openai`. |

API-backed experiments may incur charges. The following generation settings are
optional and default to off or to the values shown:

| Variable | Effect |
|---|---|
| `CAELUM_HF_USE_4BIT` | `1` forces and `0` disables 4-bit NF4 loading. When unset, model names containing 13b, 14b, 32b, 35b, 70b or larger (and Phi-4) load in 4-bit unless the repository is already quantized. |
| `CAELUM_HF_DTYPE` | `bfloat16` or `float16` overrides the Hugging Face load dtype. |
| `CAELUM_HF_DEVICE_MAP_AUTO`, `CAELUM_HF_MAX_MEMORY_GIB` | Spread a model across GPUs with `device_map="auto"` and a per-GPU memory cap. |
| `CAELUM_DISABLE_COMPILE` | `1` skips `torch.compile`. |
| `CAELUM_FP8_FORCE_TRITON` | `1` uses the Triton FP8 kernel on nodes without a CUDA toolkit. |
| `CAELUM_VLLM_TP`, `CAELUM_VLLM_PP`, `CAELUM_VLLM_GPU_UTIL`, `CAELUM_VLLM_MAX_LEN` | Tensor and pipeline parallel size (1, 1), GPU memory fraction (0.92) and context length (4096) for `--backend vllm`. |
| `CAELUM_JUDGE_REASONING_EFFORT` | Passed to the judge API, for example `none` to turn off thinking on Gemini 2.5 models. |

The Slurm launchers take their cluster paths from the environment as well; see
[Running on a Slurm cluster](#running-on-a-slurm-cluster).

## Repository layout

```text
.
├── src/
│   ├── caelum/                  # library: generation, matching, uncertainty, calibration
│   ├── run_pipeline.py          # samples answers, clusters them, scores uncertainty
│   ├── run_*.py                 # experiment and analysis entry points (src/README.md)
│   └── prepare_datasets.py      # rebuilds data/eval/ from the source datasets
├── data/eval/                   # the six benchmarks, ready to use
├── results/
│   ├── generations/             # cached answers and uncertainty per model and dataset
│   ├── llm_judged_unc_*/        # the same caches with LLM-judged match labels
│   ├── conu_table1/             # ConU baseline inputs and comparison table
│   ├── phi4_escalation/         # inputs and outputs of the Phi-4 escalation pilot
│   ├── qwen35_support_abstain/  # Qwen3.5 support-abstention pilot
│   └── *.json                   # analysis outputs behind the paper's tables and figures
├── slurm/                       # cluster launchers and environment setup (slurm/README.md)
└── docs/                        # project page, figures and asset licenses
```

`results/plots/` is created locally by the analysis scripts and is not tracked.

## Data

The six benchmarks are already prepared in `data/eval/`:

| Dataset | Questions | Type |
|---|---:|---|
| TriviaQA | 300 | factual |
| OpenNQ | 300 | factual |
| SimpleQA | 500 | factual |
| PopQA | 500 | factual |
| AmbigQA | 200 | ambiguous |
| AmbigInst | 304 | ambiguous |

This gives 2,104 questions **per generator**. `data/eval/CondAmbigQA.json` is
kept but excluded from the reported benchmark, because the heuristic matcher
accepts almost any answer against its long references.
`src/prepare_datasets.py` rebuilds the files from their sources and refuses to
replace existing files unless `--overwrite` is passed.

The cached generations in `results/generations/` cover Qwen3-8B,
Ministral-3-8B-Instruct-2512 and Gemma-4-E4B-it (the 18 main cells) and
Qwen3-14B and Qwen3-32B (12 further cells). Each `unc_*` file holds the
sampled answers, their semantic clusters, the matching label and the
uncertainty estimates for one model–dataset cell. The Qwen3-8B and Qwen3-32B samples
behind the generator-scale table are in `results/generations/fresh_scale_20261103/`,
and `results/llm_judged_unc_full/` holds the same caches with LLM-judged labels.

Dataset sources are [TriviaQA](https://github.com/mandarjoshi90/triviaqa),
[Natural Questions](https://github.com/google-research-datasets/natural-questions),
[AmbigQA](https://github.com/shmsw25/AmbigQA),
[SimpleQA](https://github.com/openai/simple-evals#evals),
[PopQA](https://github.com/AlexTMallen/adaptive-retrieval), and the
[spectral uncertainty release](https://github.com/MLO-lab/spectral_uncertainty_decomposition)
for AmbigInst and the curated evaluation inputs. Their terms are listed in
[docs/ASSET_LICENSES.md](docs/ASSET_LICENSES.md).

## Running the experiments

Run every command from the repository root. Each entry point's `--help` exits
without writing files. Recomputing an analysis from the cached answers and
drawing new samples from a model are different experiments: the commands below
recompute analyses unless stated otherwise. Analysis scripts read caches from
`results/generations/` (another folder can be set with `--generations_dir`
where offered) and write to `results/` or `results/plots/`, so use a separate
checkout to compare a rerun with the committed outputs. Add `--write-markdown`
where supported to export a local Markdown report.

### Cached analyses

Representative entry points are:

```bash
python src/run_headline_multiseed.py
python src/run_auroc_multiseed.py
python src/run_stratified_pregen.py --seeds 42 43 44 45 --all_models
python src/run_matchability_transfer.py
```

Without `--all_models`, `run_stratified_pregen.py` covers the 18 main cells.
With it, the script adds Qwen3-14B and Qwen3-32B (30 cells) and writes the
pooled values reported in the paper to
`results/stratified_pregen_all_models_summary.json`. The [source map](src/README.md)
lists every entry point by workflow. Some scripts use the 18 main cells and
others the 30-cell extension, so check the [protocol map](#protocol-map) before
combining outputs.

### Low-match regime: sampling budget, escalation and held-out families

These analyses use the committed caches and the judged labels:

```bash
python src/run_caelum_abstain.py --budget
python src/run_caelum_abstain.py --pooled --loco_scope family
D="triviaqa opennq ambiginst simpleqa popqa ambigqa"
B4="qwen3_8b ministral_3_8b_instruct_2512 gemma_4_e4b_it qwen3_14b"
python src/run_caelum_escalate.py --fallback_model qwen3_14b --datasets $D --policy switch --out_prefix caelum_escalate_full_switch
python src/run_caelum_escalate.py --fallback_model qwen3_14b --datasets $D --policy union --out_prefix caelum_escalate_full_union
python src/run_caelum_escalate.py --fallback_model qwen3_32b --datasets $D --base_models $B4 --policy switch --out_prefix caelum_escalate_32b_switch
python src/run_caelum_escalate.py --fallback_model qwen3_32b --datasets $D --base_models $B4 --policy union --out_prefix caelum_escalate_32b_union
```

They write `results/caelum_abstain_budget.json` (matchability under a larger
sampling budget), `results/caelum_abstain_pooled_family.json` (the learned
support-failure score with the target's model family held out) and
`results/caelum_escalate_*.json` (escalate-then-abstain with Qwen3-14B and
Qwen3-32B fallbacks).

### Escalation and support-abstention pilots

The two appendix pilots keep their inputs and outputs in their own folders.
The Qwen3-14B caches in `results/phi4_escalation/` come from a separate
generation run and differ from those in `results/generations/`.
`results/qwen35_support_abstain/` holds three 100-question Qwen3.5-35B-A3B
caches.

```bash
python src/run_model_escalation_rescue.py --generations_dir results/phi4_escalation --output_dir results/phi4_escalation --out_prefix model_escalation_all_to_phi4_learned_multiseed --seeds 42 43 44 45 --fallback_model phi_4 --base_models qwen3_8b gemma_4_e4b_it ministral_3_8b_instruct_2512 qwen3_14b --quick_lambda_grid
python src/run_set_escalation_rescue.py --generations_dir results/phi4_escalation --output_dir results/phi4_escalation --fast_lambda_grid
python src/run_support_abstain_trial.py --generations_dir results/qwen35_support_abstain --output_dir results/qwen35_support_abstain
```

Each command rewrites the committed output in its folder.

### ConU baseline

`slurm/conu_table1.sh` computes the per-cell similarity inputs on a GPU
(`results/conu_table1/stage1/`). The comparison then runs on a CPU:

```bash
python src/run_conu_baseline.py table1 --root results/conu_table1
```

It rewrites `results/conu_table1/conu_table1_protocol.json`, which applies the
ConU authors' scoring and set rule on the main table's split. ConU calibrates
only on questions whose reference appears among the sampled answers, so the
file also reports CAELUM and Vanilla CP calibrated on the same questions.

### Coefficient dispersion and the validation weight audit

`results/stage3_results.json` holds the main-table protocol (deterministic APS,
seed-42 40/30/30 split), and `results/stratified_lambdastar_results.json` the
integrated stratified variant (randomized APS, seed-42 25% splits). The same two
protocols on split seeds 43--45 are stored in `results/split_seeds/seed<N>/`.
`results/lambda_deterministic_multiseed.json` and
`results/lambda_integrated_multiseed.json` summarize the selected
$\lambda^\star$ of each protocol over the four seeds. Both are separate from the
randomized-APS summary in `results/headline_multiseed.json`.

```bash
for s in 43 44 45; do
  python src/aggregate_with_baselines.py --seed $s --output_dir results/split_seeds/seed$s --overwrite
  python src/run_stratified_with_lambdastar.py --seed $s --output_dir results/split_seeds/seed$s
done
python src/summarize_lambda_seeds.py
```

The weight-movement audit on the validation split $\mathcal{V}_1$ writes
`results/validation_weight_audit.json`:

```bash
python src/run_validation_weight_audit.py
```

### New generation

```bash
python src/run_pipeline.py --dataset triviaqa --model_name Qwen/Qwen3-8B --triviaqa_path data/eval/TriviaQA.json --n_samples 20 --n_clarifications 5 --n_answers 10 --uncertainty_method vne --weight_fn additive --alpha 0.1
```

This draws new samples for 20 TriviaQA questions with five clarifications and
ten answers per clarification, the budget used in the paper. `run_pipeline.py` seeds Python,
NumPy and PyTorch with `--seed` before generation and records the seed in its
log. This covers the Hugging Face backend, while vLLM and TensorRT-LLM keep
their own sampler state. The generator-scale caches were drawn with seed 42.
The caches at the top level of `results/generations/` were drawn without a fixed
sampling seed and cannot be redrawn exactly. Existing caches are reused unless
regeneration is requested.
External clarification (`--clarif_provider`) changes the pipeline and its
matchability and must be evaluated separately.

<details>
<summary>Checkpoints for the Qwen3-8B/32B support comparison</summary>

| Model | Model repository revision |
|---|---|
| Qwen/Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` |
| Qwen/Qwen3-32B | `9216db5781bf21249d130ec9da846c4624c16137` |

Generation records keep the configuration and output checksums of each run.

</details>

### Running on a Slurm cluster

The launchers in `slurm/` read their paths from the environment:

```bash
export WORK_ROOT=/path/to/project/storage   # required: environments, .env and synced results live here
export ENV_PREFIX=$WORK_ROOT/envs/caelum      # optional: Conda environment used by the launchers
export SCRATCH=/scratch/$USER                 # optional: defaults to /scratch/$USER
bash slurm/setup_env.sh --prefix "$ENV_PREFIX"
sbatch --array=0-17 slurm/full_benchmark.sh
```

`sbatch` passes the submitting shell's environment to the job, so exporting
these variables once (for example in `~/.bashrc` on the cluster) is enough.
Download the model weights once before submitting a whole array (for example,
run a single task first): several jobs downloading the same model into an empty
Hugging Face cache at the same time can corrupt the cached files.
Partitions, module names and the `/scratch/%u/caelum/logs/` output paths in the
`#SBATCH` headers follow the University of Lausanne DCSR clusters; adapt them,
and pass `--account=<account>` to `sbatch` where your site requires one. The
[cluster guide](slurm/README.md) lists every launcher and the variables the
sync and ConU scripts need.

## Protocol map

The main generators are Qwen3-8B, Ministral-3-8B-Instruct-2512 and
Gemma-4-E4B-it. Results families differ in matcher, score and split, so
similar filenames do not imply the same protocol.

| Result family | Inputs and score | Split and scope |
|---|---|---|
| Main table | Heuristic matching, deterministic APS, additive weight | Seed 42; 40% weight selection, 30% calibration, 30% test; 18 cells. |
| Multi-seed set sizes | Heuristic labels; randomized APS for CAELUM, Vanilla CP and TrueAPS, deterministic APS for SE-CP | Seeds 42–45; 40/30/30; 18 cells. |
| Learned stratum | Question features, frozen logistic classifier, heuristic labels, $\lambda=0$ weighted APS | Seeds 42–45; 30% classifier fitting, 30% calibration, 40% test; 30 cells. Learned-only results use 78 of 120 cell–seed rows from 27 cells. Oracle-fallback rows substitute true labels and are reported separately. |
| Integrated stratum | $\lambda$ selected on its own split | Seed 42; 25/25/25/25; 18 cells. |
| Transfer across generators | Dataset and question-ID alignment with reference and alias checks | Fit on the full source question pool and score the same records under source and target labels. Six directions between the main generators, 20 with Qwen3-14B/32B. The questions are shared, so this is not generalization to unseen questions. |
| Judged 14B/32B extension | A judge revisits heuristic non-matches; 30 cells | Matching, set evaluation, classifier and abstention analyses each have their own split. Auditing non-matches does not validate heuristic matches. |
| Weight sensitivity sweep | Recalibrated additive weights on a finite $\lambda$ grid, heuristic caches | 40/30/30, seeds 42–45. |
| Classifier operating points | Judged labels, question features, no oracle substitution | Seeds 42–45; 30/30/40. AUROC and AP use the 116 test folds with both classes; Brier and ECE use all 120. |
| Abstention | Judged labels, randomized APS, abstention as an extra outcome | 25/25/25/25, with the calibration quarter split between set and trigger calibration. Semantic coverage, augmented coverage and abstention cost are reported together. |
| Generator-scale support audit | New FP16 samples, five clarifications × ten answers, heuristic matching | Generation seed 42 on the same ordered questions. Reports all-question matchability only. |

In the set-size sweeps, partitions follow seeds 42–45 and CAELUM's calibrator
follows the split seed. Vanilla CP, Equal-weight, Epistemic-only and TrueAPS
keep calibrator seed 42 on each partition, and SE-CP uses deterministic APS.

## Included results

Each JSON file in `results/` is the output its script produces when run as
documented, in the environment that produced it (Windows, Python 3.11.13,
NumPy 1.26.4, SciPy 1.11.4, scikit-learn 1.6.1). A few conventions matter when
comparing a rerun:

- When a calibration subset is too small for the conformal rank, the threshold
  is infinite and every sampled cluster is kept.
- The routing-curve scripts (`run_r_vs_uepi_routing.py`,
  `r_protocol_experiment.py`) and the support-abstention pilot break score ties
  with a stable sort, which gives the same output on NumPy 1.26 and 2.x.
- Both stratification outputs exclude the duplicate 50-question Qwen3-8B
  AmbigInst cache, as the paper does.
- The transfer outputs join records by dataset and question ID, so repeated
  question texts keep their own labels. The transfer values reported in the
  paper agree with these outputs within 0.00015.
- The code is kept exactly as used for the reported results, so the stored
  outputs reproduce in the environment described above. Two details are worth
  knowing when reusing it: a reference alias made only of an emoji normalizes to
  an empty string and therefore matches any answer (six PopQA labels), and set
  construction compares cumulative mass with the threshold times the weight, so a
  cluster can be left out at an exact floating-point tie. They affect only a few
  reported values.

## Reproducibility note

On the Linux environment of `requirements.txt` (NumPy 2.4.4, SciPy 1.17.1,
scikit-learn 1.8.0), classifier- and encoder-based analyses differ slightly
from the stored outputs. Six printed values move across a rounding boundary:
the gradient-boosting row of the detection-baselines table (0.566 to 0.567,
±0.16 to ±0.15), the CAELUM detector row of the abstention-baselines table
(abstention rate 0.459 to 0.458, coverage 0.924 to 0.923), the fraction
selected at threshold 0.5 in the operating-point table (0.578 to 0.579), and
LearnedNet-RF accuracy at a 20% budget in the Phi-4 table (0.615 to 0.616).
The encoder ablation (`src/embedding_ablation.py`) moves more, because it
re-embeds and re-clusters every sampled answer: on that environment its match
rates change by up to 2.5 points, coverage by up to 0.033, AUROCs by up to 0.17,
and its selected $\lambda^\star$ widely, since that objective is nearly flat.
The paper reports the stored values.

One complete generation run over the five generators and the two pilots used
**82.13 allocated GPU-hours across 39 generation and uncertainty jobs** on the
DCSR clusters of the University of Lausanne. Each job used one GPU and 96 GB of host memory. Hardware varied
across jobs, so the timings do not compare model speed.

| Model | Jobs | NVIDIA GPU | CPU cores per job | GPU-hours |
|---|---:|---|---:|---:|
| Qwen3-8B | 6 | A100 | 8 | 6.44 |
| Ministral-3-8B-Instruct-2512 | 6 | A100 | 8 | 11.58 |
| Gemma-4-E4B-it | 6 | A100, H100 | 8 | 14.98 |
| Qwen3-14B | 6 | A100, H100, L40S | 8 | 7.88 |
| Qwen3-32B | 6 | H100 | 8 | 12.59 |
| Phi-4 | 6 | A100 | 8 | 19.57 |
| Qwen3.5-35B-A3B | 3 | H100 | 4 or 8 | 9.09 |
| **Total** | **39** | | | **82.13** |

GPU-hours are Slurm elapsed time multiplied by the allocated GPU count. They
measure allocated time, not utilization.

## Cite this work

```bibtex
@inproceedings{
azizi2026matchabilityaware,
title={Matchability-Aware Conformal Prediction for Open-Ended Language Model Generation},
author={Ilia Azizi},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=eY9UXFg5E0}
}
```

## License

The code and documentation are released under the [MIT License](LICENSE).
Third-party code, datasets and models keep their own terms, listed in
[docs/ASSET_LICENSES.md](docs/ASSET_LICENSES.md).
