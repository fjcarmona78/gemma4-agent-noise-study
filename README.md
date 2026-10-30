# Most Agent Improvements Are Noise

Code, per-task results and analysis behind the paper-track entry
*Most Agent Improvements Are Noise: A Calibrated Study of a Gemma 4 31B Coding Agent*
for the Google Gemma 4 Developer Agent competition (Kaggle, 2026).

The paper measures how much of a reported agent gain can be trusted: byte-identical Kaggle
submissions scored 0.18 then 0.13; 38 complete local runs of two different agent designs on
216 SWE-rebench tasks are statistically identical; twenty interventions moved nothing beyond
noise; and the anatomy of 995 failed attempts says why. Every number in the paper can be
rebuilt from the CSV files in `results/` with one script and no GPU.

## Rebuild the paper's tables

```bash
python analysis/reproduce_tables.py --results results
```

Standard library only. The same code runs in the companion Kaggle notebook
`analysis/reproduce_tables.ipynb`, which reads the CSV files from this repository.

## What is released

| Path | Contents |
|---|---|
| `results/tasks.csv` | The 216 distinct tasks: id, repository, creation date, which local set(s), source dataset |
| `results/runs.csv` | One row per local run (108): set, design or variant, repeat index, solves, caps, mean minutes and calls |
| `results/task_attempts.csv` | One row per (run, task) attempt (4,214): outcome, calls, minutes, how it ended, failure anatomy, file stage, first edit call, test-reading flags |
| `results/task_solve_rates.csv` | Per task: solve rate under each design over its complete runs, pooled rate and the always / coin-flip / never class |
| `results/kaggle_submissions.csv` | Our Kaggle submissions with public scores, including the byte-identical replicates |
| `results/interventions.csv` | The paper's intervention table with the run names behind each row |
| `analysis/` | `export_results.py` (builds the CSVs from raw runs), `reproduce_tables.py`, the notebook |
| `agents/safe_prompt/` | Our agent: `agent.yaml`, system prompt, sampling and evaluation config |
| `agents/variants/` | The intervention variants of our own design that ran locally (prompt and config files only; adapters not included) |
| `harness/` | Local evaluation harness: `run_eval.py` (competition harness driver), `run_rebench.py` and `run_ood45.py` (per-task Docker images for SWE-rebench and OOD-45), `serve_vllm.sh`, the queue runner |
| `tasksets/` | Task id lists of the five local sets and the held-back final-25, the repository exclusion lists, and the task-set builder scripts |
| `tools/` | Analysis tools: paired sign tests, failure anatomy, per-task tracker, trace digests, recall probe, routing check, patch chooser, repair cards, and more |
| `finetune/` | LoRA / KTO data pipeline and training scripts (data and adapters not included) |

Not included: the competition's own data (rule 2.4b forbids redistribution), the SWE-rebench
and OOD-45 task files and Docker images (public upstream, see below), raw traces, model weights
and adapters.

## The bundle

The second design of the paper, "the bundle", is the public Kaggle notebook
[Gemma 4 | Budget - Aware Coding Agent v2](https://www.kaggle.com/code/flexonafft/gemma-4-budget-aware-coding-agent-v2)
by Kaggle user flexonafft. Its files are not redistributed here. The variant used in the paper
(`public_bav2_noedit_t7` in `results/runs.csv`) differs from that notebook in three ways: `edit_file`
is removed from the tool list so that edits go through `run_command` heredocs, `max_time_minutes`
is 7 instead of 8, and the prompt's budget line says so. The unmodified bundle is Kaggle
submission 57272241 in `results/kaggle_submissions.csv`; the other `public_*` run names in
`runs.csv` are bundle variants (call caps, temperature, thinking budget, edit deadline, tool removal,
E4B), each a one-line change from the notebook's configuration as named.

## Task sets

All 216 tasks come from public sources outside the competition data.

| Set | Tasks | Source |
|---|---|---|
| V2-74 | 74 | [SWE-rebench V2](https://huggingface.co/datasets/nebius/SWE-rebench), Python tasks with validated Docker images |
| recent-33 | 33 | SWE-rebench V2, tasks created February to July 2025 |
| OOD-45 | 45 | [OOD-45](https://www.kaggle.com/datasets/rodrigospalma/ood45-bugfix-split), a leakage-audited post-release split in the competition's task format |
| fresh-15 | 15 | SWE-rebench V2, imported late as an untouched check (shares 4 tasks with V2-74, counted once) |
| rebench-dev | 53 | [SWE-rebench V1](https://huggingface.co/datasets/nebius/SWE-rebench) development split |

`tasksets/bench_repos.json` and the other exclusion lists kept every evaluation repository out
of all training data; `tasksets/builder/` holds the scripts that selected and validated the sets.

## Column notes for `task_attempts.csv`

- `design`: `bundle`, `safe`, or `variant` (then `variant` names the configuration; see `agents/variants/`).
- `run_index`: 1 for the first run of a configuration on a set, 2 for `_rep`, 3 for `_rep2`, and so on.
- `ended_by`: `time_cap` (local cap, 0.8 x the Kaggle minutes since the local GPU decodes faster), `tool_cap`, `error`, or `finished`.
- `failure_class`: `solved`, `no_patch`, `only_new_hidden_tests_fail` (only tests added or changed by the task's test patch failed), `only_existing_tests_fail`, `both_fail`, `tests_did_not_load`.
- `file_stage`: `solved`, `no_patch`, `wrong_file` (no overlap with the reference fix's source files), `right_file_test_fails`.
- `first_edit_call`, `n_edits`, `n_tests`, `read_test_file`: parsed from the run's trace with a heuristic classifier (`write_file`, or a `run_command` that writes a file; `pytest`/`unittest`/`tox`; `read_file` on a test path). Treat them as approximate.

Reference patches and tests were read for diagnosis only, never shown to an agent.

## Running the harness locally

The harness driver `harness/run_eval.py` uses the competition's own `swegemma`, `adk-submission`
and `adk-eval-core` packages, distributed as wheels in the competition's Kaggle wheelhouse
dataset; install them from there. The model is `google/gemma-4-31b-it-qat-w4a16-ct` served by vLLM
0.19.1 (`harness/serve_vllm.sh`). `harness/run_rebench.py` and `harness/run_ood45.py` wrap the
driver so that each task runs in its own upstream Docker image. `harness/runner/lib.sh` and
`queue_runner.sh` run experiments from a job queue with the same limits as the paper
(28 tool calls, 7 Kaggle minutes = 5.6 local minutes, 240-second command timeout).

## License

Apache License 2.0 (see `LICENSE`).

## Disclosure

Experiments were designed and run with help from an AI coding assistant (Claude Code, Anthropic),
which also drafted the paper and this repository's text. All numbers come from logged runs. No
outputs of proprietary models were used as training data.
