# Prompt Engineering for Cultural Image Captioning in Indigenous Languages of the Americas

Experiment code for our AmericasNLP 2026 Shared Task submission on **Cultural Image Captioning**. We run a controlled prompt-engineering study — eight prompting strategies × four VLMs (2B–32B) — for a *Spanish-pivot* captioning pipeline covering four Indigenous languages: **Wixárika, Bribri, Guaraní, and Orizaba Nahuatl**.

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
    ├── Qwen2B_COMET/                   # COMET eval job (Qwen3-VL-2B, pilot set)
    ├── Qwen3-VL-8B-Instruct/           # 8B prompt-ablation jobs
    └── Qwen3-VL-32B-Instruct/          # 32B prompt-ablation jobs
```

Each SLURM job loops over prompt strategies, calling a captioning driver, then the official repo's `translate.py`, then `eval.py`. The `baseline/downstream/` and `baseline/modal/` scripts are standalone ablations (Gemini as downstream translator; cloud-hosted VLMs via Modal) run independently of the SLURM jobs.

## Prompt strategies (`build_prompt()`)

`p0_short/medium/long_literal` (literal baseline) · `p1_culture_aware` · `p2_translation_friendly` / `p2b_..._detailed` · `p3_object_action` · `p4_direct_target` (no pivot, control)

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

(Fill in the `<LRZ_PROJECT_ID>` / `<LRZ_ACCOUNT>` placeholders for your own cluster account first.)

Requires the official repo cloned alongside this one for data + baseline MT/eval, plus a base env (`transformers`, `torch`, `pandas`, `pillow`, `tqdm`) and a separate `mt_env` for translation/eval.

## Key results

- **p0-long-literal** is the most robust cross-language baseline; **p1-culture-aware** can beat it, especially at larger scale.
- Model scale helps culture-aware prompts more than literal ones; few-shot prompting generally hurts.
- The Spanish pivot is essential — direct target-language generation scores 2–3x lower.
- The downstream translation stage can substantially affect final quality: Gemini 3 + BM25 retrieval outperforms the NLLB baseline across all four evaluated languages, while a separate in-domain Wixárika configuration with 20 translation exemplars achieves **21.53 ChrF++**.

Full tables/analysis are in the paper.

## Authors

Shengye Wu, Hamza Ben Yaacoub, Servesh Khandwe — TUM

## License

Code in this repo is MIT licensed (see `LICENSE`). The shared-task dataset itself is released separately by the organizers under CC BY-NC 4.0.
