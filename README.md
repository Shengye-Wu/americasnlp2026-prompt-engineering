# Prompt Engineering for Cultural Image Captioning in Indigenous Languages of the Americas

Experiment code for our TUM Master's practical course project based on the AmericasNLP 2026 Shared Task on **Cultural Image Captioning**. We study eight main prompt configurations across four Indigenous languages — **Wixárika, Bribri, Guaraní, and Orizaba Nahuatl** — for a *Spanish-pivot* captioning pipeline, primarily using Qwen3-VL at 2B, 8B, and 32B scales, with LLaVA-OneVision-7B and larger/API-hosted models used in additional ablations.

Builds on the official shared-task repo (datasets, baseline VLM+MT pipeline, eval scripts): https://github.com/AmericasNLP/americasnlp2026

TUM Master's practical course project, supervised by Shu Okabe.

## Pipeline

```
Image ──► VLM + prompt ──► Spanish caption ──► Translation model ──► Target-language caption
```

A VLM writes a Spanish caption; a downstream MT model (baseline: Sheffield NLLB; ablation: Gemini 3 + retrieval) translates it into the target language.

## Repository structure

```
.
├── baseline/
│   ├── captioning/     # VLM → Spanish caption drivers (Qwen3-VL, LLaVA-OneVision)
│   ├── downstream/     # Gemini-as-translator ablations (few-shot / BM25-retrieval), vs. NLLB
│   └── modal/          # Modal cloud jobs: few-shot captioning, OpenRouter VLMs, full pipeline
└── slurm/
    ├── Qwen3-VL-2B-Instruct/           # 2B full-grid sweep scripts (plain bash, all languages)
    ├── Qwen2B_COMET/                   # COMET eval job (Qwen3-VL-2B, pilot set)
    ├── Qwen3-VL-8B-Instruct/           # 8B prompt-ablation jobs
    └── Qwen3-VL-32B-Instruct/          # 32B prompt-ablation jobs
```

Each SLURM job loops over prompt strategies, calling a captioning driver, then the official repo's `translate.py`, then `eval.py`. The 8B and 32B jobs are `.sbatch` files sliced by language/prompt to stay inside cluster walltime limits; the `Qwen3-VL-2B-Instruct/` scripts are plain `bash` drivers that sweep the whole grid (all four languages, all prompt modes) in one run, since the 2B model is fast enough not to need slicing. The `baseline/downstream/` and `baseline/modal/` scripts are standalone ablations (Gemini as downstream translator; cloud-hosted VLMs via Modal) run independently of the SLURM jobs.

## Prompt strategies (`build_prompt()`)

`p0_simple` · `p0_short/medium/long_literal` (literal baseline) · `p1_culture_aware` · `p2_translation_friendly` / `p2b_..._detailed` · `p3_object_action` · `p4_direct_target` (no pivot, control)

Additional experiment: `p5_few_shot` using 5 or 20 multimodal Wixárika pilot examples.

## Usage

```bash
python baseline/captioning/run_qwen_prompt_experiment.py \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --language wixarika --split dev \
  --prompt-mode p0_long_literal \
  --max-samples 50 --max-new-tokens 96 \
  --output-prefix baseline/output/wixarika_8b_p0_long_50
```

Or submit a full pipeline job:

```bash
sbatch slurm/Qwen3-VL-8B-Instruct/run_qwen8b_p0_short_medium_long_p1_p2_p3_p4_wixarika.sbatch
```

Or run the 2B grid sweep directly (from the official repo root):

```bash
bash slurm/Qwen3-VL-2B-Instruct/run_qwen2b_all_langs.sh          # p0_long_literal, all 4 languages
bash slurm/Qwen3-VL-2B-Instruct/run_qwen2b_full_poster_tests.sh  # every prompt mode × all 4 languages, + poster CSV summaries
```

## Installation and setup

This repository extends the official [AmericasNLP 2026 repository](https://github.com/AmericasNLP/americasnlp2026) and is not intended to be run as a completely standalone project.

### 1. Set up the official AmericasNLP 2026 baseline

First, clone the official repository and follow its installation instructions:

```bash
git clone https://github.com/AmericasNLP/americasnlp2026.git
cd americasnlp2026
```

Before running our experiments, make sure that the official baseline pipeline works correctly, in particular the Sheffield NLLB-based translation component. Our main experiments reuse the baseline data layout, translation code, and evaluation pipeline provided by the official repository.

### 2. Add the experiment scripts from this repository

Clone this repository separately:

```bash
git clone https://github.com/Shengye-Wu/americasnlp2026-prompt-engineering.git
```

The experiment files should then be copied into the official `americasnlp2026` repository:

- Copy the contents of `americasnlp2026-prompt-engineering/baseline/` into `americasnlp2026/baseline/`.
- Copy `americasnlp2026-prompt-engineering/slurm/` into the root of `americasnlp2026/`.

Afterwards, the relevant directory structure should look approximately like:

```
americasnlp2026/
├── baseline/
│   ├── americasnlp-2023-sheffield/    # official NLLB translation component
│   ├── captioning/                     # Qwen3-VL captioning experiments
│   ├── downstream/                     # Gemini downstream translation experiments
│   └── modal/                          # Modal/OpenRouter experiments
├── data/
├── slurm/
│   ├── Qwen3-VL-2B-Instruct/
│   ├── Qwen3-VL-8B-Instruct/
│   ├── Qwen3-VL-32B-Instruct/
│   └── Qwen2B_COMET/
└── ...
```

Run the experiment commands from the root of the official `americasnlp2026` repository.

For SLURM experiments, create the log directory if necessary:

```bash
mkdir -p logs
```

Cluster-specific placeholders such as `<LRZ_PROJECT_ID>` and `<LRZ_ACCOUNT>` in the SLURM scripts must be replaced with the corresponding values for your own computing environment.

### 3. Install the Python environments

- **Base env** (captioning, downstream ablations, Modal orchestration): `pip install -r requirements.txt` (from this repo).
- **`mt_env`** (translation + evaluation): a separate environment for the official repo's `translate.py`/`eval.py` — install its own `requirements.txt`, plus `unbabel-comet` if you're also running the `Qwen2B_COMET` slurm job.

### 4. (Optional) API credentials for the hosted-model ablations

See [API-based experiments](#api-based-experiments) below if you plan to run the OpenRouter/Modal scripts.

## API-based experiments

Some supplementary experiments use hosted models through OpenRouter/Modal and therefore require the corresponding API credentials. API keys are not included in this repository and should be provided through environment variables or the platform's secret-management mechanism.

- `baseline/downstream/gemini_downstream.py`, `gemini_rag_downstream.py`, and the Modal scripts that call OpenRouter (`modal_openrouter.py`, `modal_fewshot.py`) all read the key from the `OPENROUTER_API_KEY` environment variable.
- For the local downstream scripts, export it directly, e.g. `OPENROUTER_API_KEY=$(cat ~/.openrouter_key) python baseline/downstream/gemini_downstream.py ...`.
- For the Modal scripts, it's injected via a named Modal secret instead: `modal secret create openrouter-key OPENROUTER_API_KEY=...` (once per account), which also requires `modal setup` to authenticate the Modal CLI itself.

## Key results

- **p0-long-literal** is the most robust cross-language baseline; **p1-culture-aware** can beat it, especially at larger scale.
- Model scale helps culture-aware prompts more than literal ones; few-shot prompting generally hurts.
- The Spanish-pivot pipeline substantially outperforms direct target-language generation in our evaluated settings.
- The downstream translation stage can substantially affect final quality: Gemini 3 + BM25 retrieval outperforms the NLLB baseline across all four evaluated languages, while a separate in-domain Wixárika configuration with 20 translation exemplars achieves **21.53 ChrF++**.

Full tables/analysis are in the paper.

## Authors

Shengye Wu, Hamza Ben Yaacoub, Servesh Khandwe — TUM

## License

Code in this repo is MIT licensed (see `LICENSE`), **except** `baseline/captioning/run_qwen_prompt_experiment.py` and `run_prompt_experiment.py`, which are modified from the AmericasNLP 2026 organizers' `baseline/caption_generation.py`. That upstream file carries no stated license — the official repo's only license statement covers the dataset, not its code — so those two files are carved out of the MIT grant pending clarification/permission from the organizers. The shared-task dataset itself is released separately under CC BY-NC 4.0.
