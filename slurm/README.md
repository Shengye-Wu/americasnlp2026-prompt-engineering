# SLURM Job Scripts

Batch scripts for running the caption → translate → evaluate pipeline on the LRZ cluster.

## Why are there so many scripts?

The full experiment grid — prompt strategies × languages × model sizes — takes too long to run as a single SLURM job: each prompt mode over 50 dev images can take a couple of hours per language, and the cluster partitions cap job walltime (`#SBATCH --time=...`, e.g. 18h for the 8B jobs, 12h for the 32B jobs). So instead of one script that loops over everything, each script covers a slice of the grid — one or more prompt modes for one language. This keeps individual jobs inside the time limit and lets a failed or timed-out slice be resubmitted on its own instead of rerunning the whole matrix.

## Directory layout

- `Qwen3-VL-8B-Instruct/` — 8B model. Filenames encode every prompt mode the script runs: `run_qwen8b_<prompt modes>_<language>.sbatch` (e.g. `run_qwen8b_p1_p2_wixarika.sbatch` runs `p1_culture_aware` then `p2_translation_friendly`).
- `Qwen3-VL-32B-Instruct/` — 32B model, one prompt mode per language: `run_qwen32_<prompt mode>_<language>.sbatch`.
- `Qwen2B_COMET/` — a single job that runs the 2B model over the *pilot* set and scores the generated Spanish captions with COMET (a sanity check on captioning quality alone, not part of the main Spanish→target pipeline).

## Editing a script

Every script follows the same structure: a block of static variables, a `PROMPT_LIST` array, then a loop that runs captioning → translation → evaluation for each entry in the list.

### Switching the model

```bash
MODEL="Qwen/Qwen3-VL-8B-Instruct"
```

Set this to any Hugging Face VLM checkpoint compatible with `baseline/captioning/run_qwen_prompt_experiment.py`.

> Note: the 32B and `Qwen2B_COMET` scripts still name this variable `MODEL32` rather than `MODEL` — same purpose, just an older variable name that hasn't been renamed there yet.

### Experiment parameters

```bash
EXP_LANG="guarani"
EXP_SPLIT="dev"
EXP_SAMPLES="50"
EXP_MODEL_ALIAS="8b"
SRC_CODE="spa_Latn"
TGT_CODE="grn_Latn"
```

| Variable | Meaning |
|---|---|
| `EXP_LANG` | Target Indigenous language: `wixarika`, `bribri`, `guarani`, or `nahuatl`. |
| `EXP_SPLIT` | Dataset split to run on: `dev` (50 images) or `pilot` (20 images). |
| `EXP_SAMPLES` | Number of images to caption — must match the split size (`50` for dev, `20` for pilot). |
| `EXP_MODEL_ALIAS` | Short tag (`8b`, `32b`, `2b`, ...) used only to build human-readable output filenames — it has no effect on which model actually runs (that's `MODEL`/`MODEL32`). |
| `SRC_CODE` | Source language code for the MT step. Always `spa_Latn` — Spanish is the pivot language every caption goes through. |
| `TGT_CODE` | Target language code for the MT step. Must match `EXP_LANG`: `wixarika` → `hch_Latn`, `bribri` → `bzd_Latn`, `guarani` → `grn_Latn`, `nahuatl` → `nah_Latn`. |

To run a different language, change `EXP_LANG` and `TGT_CODE` together.

### Prompt modes

```bash
PROMPT_LIST=(
    "p0_long_literal"
)
```

List one or more prompt-mode identifiers here; the script loops over each one, running the full pipeline per entry and printing a summary of all scores at the end. Available modes (see `build_prompt()` in `baseline/captioning/`): `p0_short_literal`, `p0_medium_literal`, `p0_long_literal`, `p1_culture_aware`, `p2_translation_friendly`, `p2b_translation_friendly_detailed`, `p3_object_action`, `p4_direct_target`.

## COMET experiments

The supplementary COMET experiments require an additional COMET installation. COMET is not required for the main ChrF++ experiments. Please install COMET following the instructions in the [upstream COMET repository](https://github.com/Unbabel/COMET) before running the COMET evaluation scripts.

The corresponding jobs are located under `Qwen2B_COMET/`.
