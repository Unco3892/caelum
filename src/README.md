# CAELUM source map

Run commands from the repository root. The reusable implementation is in
`caelum/`; the 36 Python files beside it provide experiment entry points and
shared analysis helpers. Keeping them together preserves imports and Slurm
commands.

## Commands and helpers

| Purpose | Files |
|---|---|
| Generation and evaluation inputs | `run_pipeline.py`, `prepare_datasets.py` |
| Headline and matched-coverage analysis | `aggregate_with_baselines.py`, `summarize_lambda_seeds.py`, `run_headline_multiseed.py`, `run_ncapss_full.py` |
| Table export for the 18-cell benchmark | `make_paper_tables.py` |
| AUROC and selective answering | `run_auroc_multiseed.py`, `run_delong_aurocs.py`, `r_protocol_experiment.py`, `run_r_vs_uepi_routing.py` |
| Encoder and decomposition analyses | `embedding_ablation.py`, `run_decomp_interpret.py` |
| Weight diagnostics and inverse-weight control | `run_lambda_mechanism.py`, `run_validation_weight_audit.py`, `run_inverse_multiseed.py` |
| Question-based matchability strata | `run_stratified_pregen.py`, `run_stratified_with_lambdastar.py`, `run_operating_point_audit.py`, `run_popularity_probe.py` |
| Transfer across generators | `run_matchability_transfer.py`, `run_transfer_bootstrap.py` |
| Judged matching and agreement between the two judges | `run_llm_judge_rematch.py`, `run_judge_agreement.py`, `llm_judge_matcher.py` (shared API helper) |
| Support-failure ranking on the Qwen3-14B and Qwen3-32B judged caches | `run_scale_ranking.py` |
| Qwen3-8B/32B support audit (generator-scale table) | `run_fresh_scale_support.py` |
| Judged-label analysis and abstention comparisons | `run_judged_rerun.py`, `run_caelum_abstain.py`, `run_published_baselines.py`, `run_pke_baseline.py` |
| ConU baseline on the Table 1 cells (with `slurm/conu_table1.sh`) | `run_conu_baseline.py` |
| Escalate-then-abstain with Qwen3-14B and Qwen3-32B fallbacks | `run_caelum_escalate.py` |
| Model escalation and support-abstention pilots | `run_model_escalation_rescue.py`, `run_set_escalation_rescue.py`, `run_support_abstain_trial.py` |

`make_paper_tables.py` exports four tables for the three-model, 18-cell
benchmark in a plain layout. Its labels can differ from the paper (it labels the
self-consistency proxy "ConU"), and it does not produce every reported table
and figure.

The [protocol map](../README.md#protocol-map) distinguishes the main
benchmark, larger-model analyses, judged matching, and abstention experiments.
A similar filename does not imply the same score, split, or matching rule.

## Shared implementation

The `caelum/` package contains generation, semantic matching, uncertainty,
calibration, evaluation, abstention, and baseline implementations.
`predictive_kernel_entropy.py` is a separate comparison method.
`transfer_alignment.py` aligns stable dataset and record IDs and checks reference
metadata. `plotting.py` supplies shared plotting defaults.

Some entry points also supply helpers to other analyses:

- `run_judged_rerun.py` supplies cache discovery and loading helpers.
- `run_stratified_pregen.py` supplies question features used in transfer and
  integrated-weight stratification.
- `run_matchability_transfer.py` supplies helpers to the bootstrap analysis.
- `run_headline_multiseed.py` and `run_lambda_mechanism.py` supply helpers to the
  inverse-weight analysis.
- `run_pke_baseline.py` is imported by the abstention analysis.
- `run_model_escalation_rescue.py` supplies helpers to the set-escalation pilot.

## Inputs and outputs

Canonical evaluation inputs live in `data/eval/`. Dataset preparation refuses to
replace existing files unless `--overwrite` is supplied.

`caelum/paths.py` defines the folders every script uses. Cached generations
(`gen_*`, `unc_*`) and their generation logs live in `results/generations/`, and
the LLM-judged caches in `results/llm_judged_unc_full/`. Analysis outputs go to
`results/`. Scripts that read caches accept `--generations_dir` to use another
folder, such as one of the pilot folders. Plot exports go to `results/plots/`,
and table exports go to `results/exported_tables/`; both export
directories are gitignored.

Markdown reports are optional and disabled by default. Add `--write-markdown`
to a report-producing command to save one. For `embedding_ablation.py` and
`r_protocol_experiment.py`, an explicit `--output_md PATH` also enables the
report. Markdown files under `results/` are gitignored.

The integrated-weight stratification runner keeps the largest question cache
per model/dataset cell, records excluded duplicate caches in its JSON output,
and reports learned-only aggregates separately from oracle-fallback rows.

Some analysis commands replace existing summaries. Use an isolated checkout or
an explicit alternate output path where supported when comparing a rerun with
stored results. Cached analyses can still require pretrained encoder weights.
New generation and API judging are separate experiments.

## Cluster entry points

The generation launchers call `src/run_pipeline.py`. The frontier-pilot
postprocessor calls `src/run_support_abstain_trial.py`. See the
[cluster guide](../slurm/README.md) for site settings and resource requests.
