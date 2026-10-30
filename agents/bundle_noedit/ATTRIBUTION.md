# Attribution

The files in this directory are a modified copy of the public Kaggle notebook
"Gemma 4 | Budget - Aware Coding Agent v2" by Kaggle user flexonafft
(https://www.kaggle.com/code/flexonafft/gemma-4-budget-aware-coding-agent-v2),
published for the Gemma 4 Developer Agent competition. The agent definition, sampling
configuration and the system prompt are theirs; the paper calls this design "the bundle".

Our changes (the `public_bav2_noedit_t7` variant used in the paper):

- `edit_file` removed from the tool list, so edits go through `run_command` heredocs;
- `max_time_minutes` 8 -> 7 in `eval_config.yaml`, for margin under the scorer's 12-hour limit;
- the prompt's budget line updated to match (28 tool calls, 7 minutes).

The unmodified bundle is the Kaggle submission 57272241 in `results/kaggle_submissions.csv`.
