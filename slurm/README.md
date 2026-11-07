# Cluster scripts

These scripts record the project's cluster workflows. Paths come from
environment variables, while partitions, CUDA modules, software stacks and GPU
requests follow the University of Lausanne DCSR clusters. Read a script and
adapt those settings before submitting it.

## Configuration

| Variable | Used by | Meaning |
|---|---|---|
| `WORK_ROOT` | all launchers, `setup_env.sh`, sync scripts | Project storage on the cluster (required). Environments, `.env` and synced results live here. |
| `ENV_PREFIX` | launchers | Conda environment to activate, default `$WORK_ROOT/envs/caelum`. |
| `VLLM_ENV_PREFIX` | `build_vllm_env.sh`, `gpt_oss_120b_cell.sh` | Separate vLLM environment, default `$WORK_ROOT/envs/caelum_vllm`. |
| `SCRATCH` | launchers | Scratch root, default `/scratch/$USER`. The checkout is expected at `$SCRATCH/caelum`. |
| `REMOTE`, `DEST_BASE` | `sync_to_cluster.sh`, `sync_from_cluster.sh` | `user@login-host` and the checkout directory on the cluster, for example `/scratch/<user>/caelum` (required). |
| `STS_CACHE`, `BASEPY` | `conu_table1.sh` | Hugging Face cache holding `cross-encoder/stsb-roberta-large`, and the environment's `python` (required). |

`sbatch` passes the submitting shell's environment to the job, so exporting
these once (for example in `~/.bashrc` on the cluster) is enough:

```bash
export WORK_ROOT=/path/to/project/storage
sbatch --array=0-17 slurm/full_benchmark.sh
```

`#SBATCH` headers cannot read environment variables. Job logs go to
`/scratch/%u/caelum/logs/`, so create that folder first or override it with
`sbatch --output`/`--error`, and pass `--account=<account>` where your site
requires one. The launchers export `HF_TOKEN` and the API keys from
`$WORK_ROOT/.env` when that file exists.

## Launchers

| Scripts | Purpose |
|---|---|
| `full_benchmark.sh` | Three generators × six datasets, with up to five split seeds. The array has 18 tasks per seed and 90 tasks in total. Generation caches are shared across seeds. |
| `qwen3_14b_cell.sh`, `qwen3_32b_cell.sh` | Single-cell scale experiments. |
| `qwen3_235b_cell.sh`, `gpt_oss_120b_cell.sh`, `llama33_70b_cell.sh` | Larger-model generation configurations. A launcher file does not establish a completed result. |
| `gemini_pro_clarif.sh` | External-clarifier experiment, separate from the paper's self-clarification protocol. |
| `frontier_answer_pilot.sh`, `frontier_answer_quick_signal.sh` | Frontier-generator pilots. |
| `postprocess_frontier_pilot.sh` | Support-abstention analysis from pilot caches. |
| `conu_table1.sh` | ConU similarity inputs for the 18 main-table cells, one GPU task per cell. The comparison itself runs locally with `src/run_conu_baseline.py table1`. |
| `smoke_test.sh` | Small generation and evaluation check. |

## Environments and data movement

`setup_env.sh` creates a new Python 3.11.15 / pip 26.0.1 environment and installs
[`requirements-dcsr.lock.txt`](requirements-dcsr.lock.txt). The lock records
the environment used for the generation runs. Records for the 30 main-grid generation jobs confirm
PyTorch 2.11.0+cu130 and Transformers 5.7.0 at runtime; the lock pins the PyTorch
distribution as `torch==2.11.0`. Other package versions were captured from the same
environment and are not independently confirmed for every job. A new
environment built by `setup_env.sh` (in a separate `/scratch` prefix) installed
cleanly, passed `pip check`, reported PyTorch 2.11.0+cu130 with CUDA available on
an A100, ran every entry point's `--help`, and completed a short generation run.

```bash
bash slurm/setup_env.sh --help
WORK_ROOT=/path/to/project bash slurm/setup_env.sh --prefix /path/to/new/environment
```

The default prefix is `$ENV_PREFIX`, or `$WORK_ROOT/envs/caelum` when that is
unset, so the launchers find the new environment without further settings.
Existing targets are refused. Setup leaves other environments and `~/.condarc` unchanged, uses
`CONDA_PKGS_DIRS` for its process, and stops on installation or dependency-check
failure. GPU compatibility and model
weights need separate checks. Setup does not log in, save tokens, download model
weights, or build FlashAttention. `prefetch_qwen36.sh` downloads weights separately.

`build_vllm_env.sh` is an optional helper with unpinned packages for
the vLLM launchers. It is separate from the recorded environment and
requires review before reuse.

`sync_to_cluster.sh` copies source, dependency files, evaluation data, Slurm
scripts and a local `.env` file to `$REMOTE:$DEST_BASE`, and the `.env` file
also to `$WORK_ROOT`. `sync_from_cluster.sh` writes downloaded
results, plots and logs into this checkout. Review their source and destination
paths before use, and keep a separate copy of any result snapshot being compared.

## Generation runs

The reported generation runs used separate orchestration with isolated source
snapshots and run manifests. That orchestration is not included in this
directory. The launchers listed above should not be treated as its replacement.

Check a new job's status from its logs, output files and scheduler state. The
summaries already in `results/` cannot establish that a new job has completed. The
[protocol map](../README.md#protocol-map) describes the
verified support audit separately.
