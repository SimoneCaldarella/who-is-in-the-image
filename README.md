# Who is in the Image?

**Benchmarking Any Identity Anonymization in Multimodal Large Language Models**  
Simone Caldarella · Matteo Farina · Massimiliano Mancini · Elisa Ricci · Rahaf Aljundi  
**WACV 2027 — Dataset Track**

[Paper](docs/assets/who-is-in-the-image.pdf) · **Dataset: [download link to be added](DATASET_URL_TO_BE_ADDED)** · arXiv: forthcoming

TUTUS evaluates identity-name suppression without a predefined forget set, while measuring descriptive utility and robustness to malicious prompts. The experiments cover celebrity and fictitious identities, six LLaVA/Qwen models, four training-free baselines, and four post-training objectives applied to three model components.

This repository contains the runtime code and settings needed to repeat the experiments. Historical run outputs, cluster launchers, Slurm files, notebooks, checkpoints, and dataset images are kept outside Git. The project website lives in `docs/`.

## 1. Installation

Use **Python 3.11 on Linux with an NVIDIA GPU**. The original environment used PyTorch 2.3.0, CUDA 12.1, Transformers 4.53.2, PEFT 0.16.0, and eight GPUs. Dependencies are pinned because the project includes custom Transformer model implementations. One GPU is supported subject to memory, with the total training batch maintained through gradient accumulation.

```bash
git clone https://github.com/SimoneCaldarella/who-is-in-the-image.git
cd who-is-in-the-image
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py --help
```

Install a CUDA-capable PyTorch build suitable for your machine if the default wheel is unsuitable. Model weights are fetched from Hugging Face on first use; existing caches work as usual. `--model-path /path/to/model` selects a local foundation-model directory. Keep the short `--model` identifier as well, since it selects the processor and architecture.

The small vendored `lmms-eval` subset is loaded automatically for utility evaluation. Do **not** install a separate upstream harness over it: its dependency constraints and model adapters differ from this snapshot. `decord` is a platform-specific dependency used by the retained utility adapters.

## 2. Download TUTUS

**Dataset URL: `DATASET_URL_TO_BE_ADDED`**

Download **`TUTUS-dataset.zip`**, then extract it outside the repository or under ignored `data/`:

```bash
mkdir -p data
unzip /path/to/TUTUS-dataset.zip -d data
export TUTUS_DATA_ROOT="$PWD/data/TUTUS-data"
python run.py check-data
```

Alternatively, pass `--data-root /absolute/path/to/TUTUS-data` to each command. Point to the **`TUTUS-data` directory**, not its parent and not the ZIP.

Expected layout:

```text
TUTUS-data/
  celebdata/
    train/                  # 15,000 collected preference images
    test/                   # 24,999 celebrity images
    annotations/            # Development/test prompts, preferences, original caches
  mllmu/
    images/                 # 500 fictitious identity images
    annotations/
    source/ft_Data/          # Embedded-image parquet used for identity learning
    source/Full_Set/         # Original personalization evaluation data
    models/                 # Existing personalization adapters
  provenance/original_annotations/
  DATASET_CARD.md
  DATASET_STATS.json
  MANIFEST.json
```

The preference CSVs contain multiple prompt rows per image. Training uses model-specific `TUTUS-PD-V1-<model>.csv` files and deduplicates images by default. The retained identities/images therefore vary by model; 15,000 is the collected pool size, not every model's training size.

### Which split should I use?

| CLI split | Annotation | Purpose |
| --- | --- | --- |
| `dev` | `TUTUS-DEV.csv` | 300 development images, 148 identities; tuning and checkpoint selection |
| `celebrity` | `TUTUS-TS-P.csv` | Historical celebrity annotation used by the archived experiment workflow |
| `celebrity-heldout` | `TUTUS-TS-P-HELDOUT.csv` | Explicitly excludes all development images; 24,699 images |
| `fictitious` | `MLLMU-TS-P.csv` | 500 fictitious identities, after personalization |

The historical celebrity annotation contains the 300 development images. The release preserves it and provides the explicitly disjoint alternative separately. Do not present results from a newly chosen split as identical to the manuscript's historical results. Existing caches may describe older development subsets; new runs generate caches in `runs/pristine/` rather than silently reusing them. Known broken images are excluded by the loader.

### Dataset provenance

Celebrity evaluation imagery is reused from **PHANTOM**; fictitious identities and personalization source data come from **MLLMU-Bench**. Preference images were additionally collected, with model-specific rejected outputs and preferred non-identifying responses. The archive includes attribution, original annotations, statistics, and per-file SHA-256 checksums. Upstream datasets, imagery, model weights, and the face detector retain their source terms.

Independent utility datasets—MME, MMMU, MMBench, SEED-Bench, and VMCBench—and foundation-model weights are **not** bundled. The utility harness downloads them separately. Existing preference data does not require regenerating GPT-4.1 responses. The MMBench evaluator may require `OPENAI_API_KEY` for its answer-matching fallback; other local experiments do not require an OpenAI key.

## 3. Supported configurations

```bash
python run.py list
```

| Model identifier | Foundation model |
| --- | --- |
| `llava_next` | `llava-hf/llava-v1.6-mistral-7b-hf` |
| `llava_onevision_smol` | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` |
| `llava_onevision` | `llava-hf/llava-onevision-qwen2-7b-ov-hf` |
| `qwen2vl` | `Qwen/Qwen2-VL-7B-Instruct` |
| `qwen2_5vl_smol` | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `qwen2_5vl` | `Qwen/Qwen2.5-VL-7B-Instruct` |

Training-free methods: `prompting`, `blurring`, `noise`, `pruning`.  
Post-training components: `vision`, `connector`, `text`.  
Objectives: `dpo`, `sft`, `ga`, `npo`.

`configs/paper.json` stores all **72 selected model × component × objective configurations**, their actual learning rates and selected checkpoints. Shared settings and ablation grids are also retained. Default selected checkpoints are configuration-specific; `--checkpoint perc_20` overrides the selection explicitly. Method implementations preserve the original effective LoRA settings (`r=16`, `alpha=16`, dropout `0.05`). The manuscript's separately stated dropout value was not changed during packaging.

## 4. Training-free experiments

Face blurring needs the face detector once:

```bash
python run.py download-detector
python run.py evaluate --model qwen2_5vl --method blurring --split celebrity
python run.py evaluate --model qwen2_5vl --method prompting --split celebrity
python run.py evaluate --model qwen2_5vl --method noise --split celebrity
python run.py evaluate --model qwen2_5vl --method pruning --split celebrity
```

Supply an existing detector with `--detector /path/to/yolov8m-face.pt`. The default is `assets/yolov8m-face.pt`, excluded from Git.

The CLI first creates full-split pristine predictions if they are absent, then performs the requested experiment automatically. There is no need to rerun the same command manually. Use `--dry-run` to print commands without loading models, writing run files, or generating predictions.

```bash
python run.py evaluate --model qwen2_5vl --method noise --split dev \
  --set method.level=5.0 --experiment runs/ablations/qwen-noise-5
python run.py evaluate --model qwen2_5vl --method pruning --split dev \
  --set method.sparsity=0.7
```

The paper's noise grid is `[0.5, 1, 2, 5, 10, 100]`; the pruning grid is `0.1` through `1.0`. Default noise is `2.0`, and default pruning sparsity is `0.7`.

## 5. Post-training and seen/unseen evaluation

```bash
python run.py train --model qwen2_5vl --method vision --objective dpo --gpus 8
python run.py evaluate --model qwen2_5vl --method vision --objective dpo \
  --split celebrity --gpus 8
```

Training defaults to development evaluation and uses the selected paper learning rate. It retains the original optimizer, scheduler, total batch size and seed, and saves checkpoints at 10%, 20%, 30%, 40%, 50%, 60%, 80%, and 100% of the training schedule. The total batch must be divisible by `--gpus × per-GPU batch size`.

The example writes its model/configuration/identity list under:

```text
runs/post-training/qwen2_5vl-vision-dpo/
  config.yaml
  training_classes.json
  perc_10/mypeft/
  ...
  perc_100/mypeft/
```

Seen/unseen membership follows the identities actually retained in the model-specific training data, recorded in `training_classes.json`. Evaluate a particular checkpoint on development data with:

```bash
python run.py evaluate --model qwen2_5vl --method vision --objective dpo \
  --split dev --checkpoint perc_20
```

To use anonymization weights trained or downloaded elsewhere, pass the experiment directory containing `training_classes.json` and checkpoint subdirectories:

```bash
python run.py evaluate --model qwen2_5vl --method vision --objective dpo \
  --experiment /path/to/experiment --checkpoint perc_20
```

No anonymization checkpoints are committed to this minimal repository. Train them with the commands above, or use the corresponding experiment directories from the separate full archival code ZIP. Existing checkpoints are protected against accidental retraining; choose a new `--output` or `--experiment` for another run. **No manuscript result is recalculated or edited by these commands.**

## 6. Continual adaptation and fictitious identities

The dataset archive already contains the personalization adapters used by the archived workflow. Apply the celebrity-trained anonymization adapter to the personalized model:

```bash
python run.py transfer --model qwen2_5vl --method vision --objective dpo
python run.py evaluate --model qwen2_5vl --method blurring --split fictitious
```

To repeat identity learning instead:

```bash
python run.py personalize --model qwen2_5vl --gpus 8
```

The command uses the archived source parquet. Model-specific defaults reproduce the selected schedules: LLaVA-NeXT uses 7 epochs at `2e-5`; the other five models use 10 epochs at `3e-5`. Override with `--epochs` and `--learning-rate` if deliberately running an ablation. Pass the resulting directory containing `mypeft/` to subsequent evaluations:

```bash
python run.py transfer --model qwen2_5vl --method vision --objective dpo \
  --personalization-adapter /absolute/path/to/personalization-run
```

Personalization transfers previously learned anonymization adapters; it does not train a new anonymization adapter on the fictitious identities.

## 7. Attack and benign-prompt analyses

```bash
python run.py attacks --model qwen2_5vl --method vision --objective dpo
```

This runs the fixed attack collection (direct, name-only, authority framing, step-by-step elicitation, and the simulated two-turn proxy), plus the action/clothing/scene/expression/visible-attribute benign prompts. It compares aligned and pristine responses. Default stress-test sampling is 100 identities and at most two images per identity, before model-specific eligibility filtering. Use repeatable `--set` overrides for explicit variations:

```bash
python run.py attacks --model qwen2_5vl --method vision --objective dpo \
  --set dataset.test.num_unique_classes=-1 \
  --set dataset.test.num_samples_per_class=-1
```

These controlled prompts are not a comprehensive adaptive or genuine multi-turn adversarial evaluation.

## 8. General multimodal utility

```bash
python run.py utility --model qwen2_5vl --method pristine
python run.py utility --model qwen2_5vl --method blurring
python run.py utility --model qwen2_5vl --method vision --objective dpo
```

Default tasks are `mme,mmmu_val,mmbench_en_dev,seedbench,vmcbench`. Select a subset or perform a small smoke test with:

```bash
python run.py utility --model qwen2_5vl --method pristine \
  --benchmarks mme --limit 10
```

Use complete datasets for reported results. Hugging Face authentication may be required by a source dataset. The custom adapters load the exact anonymization method from the generated configuration; the baseline `pristine` command does not apply an anonymization method.

## 9. Run complete experiment grids

A suite **prints its plan by default**. Add `--execute` to run it serially. No scheduler or Slurm installation is needed.

```bash
python run.py suite --suite training-free
python run.py suite --suite post-training --gpus 8
python run.py suite --suite transfer
python run.py suite --suite attacks
python run.py suite --suite utility
python run.py suite --suite noise-ablation
python run.py suite --suite pruning-ablation
```

The post-training suite trains all 72 configurations, evaluates the saved checkpoints on development data, and evaluates the selected checkpoints on the historical celebrity annotation. The training-free suite covers both celebrity and fictitious identities. The transfer and post-training utility suites require anonymization checkpoints to exist first. Restrict the grid with `--models qwen2_5vl`, or plan everything with `--suite all`. Full suites can take substantial GPU time; they are not launched by setup or by a dry run.

```bash
python run.py suite --suite post-training --models qwen2_5vl --gpus 8 --execute
```

## 10. Metrics, reports, and outputs

```bash
python run.py report --output runs
python run.py score \
  --pristine runs/pristine/TUTUS-TS-P_qwen2_5vl_pristine.csv \
  --predictions /path/to/vision__TUTUS-TS-P.csv \
  --save runs/leakage_counts.json
```

`report` exports `runs/report/metrics.csv`, `seen_unseen.csv`, `utility.csv`, and available attack/benign summaries. It computes the standard harmonic mean from the three component metrics, rather than trusting a historical incorrect HM implementation. **The paper's HM values were already corrected by the authors; this does not change the manuscript.** Run outputs and generated reports stay under ignored `runs/`.

RSS is conditional on examples where the base model leaks the identity. CDP uses the original sentence-transformer similarity rule. JDR measures relative degradation and is clipped at 1; consequently `1 − RSS × JDR` is a conditional leakage summary and can upper-bound observed malicious leakage. The CPU-only `score` command reports explicit coverage, eligible counts, observed conditional leakage, and image-level Wilson intervals. `--eligible shared_paths.json` restricts it to a common image set. Those intervals do not account for clustering by identity.

All scientific values remain on the `[0,1]` scale in raw metric outputs. Selected run settings and output files make the configuration traceable; stochastic training, GPU topology, software changes, and historical split provenance can affect exact numerical reproduction.

## Repository layout

```text
run.py                       # One public experiment CLI
requirements.txt             # Pinned runtime dependencies
configs/paper.json           # Selected schedules and ablation grids
tutus/
  dataset/                   # Dataset loaders
  modeling/                  # Model implementations, LoRA, noise, pruning, blur
  methods/                   # Training-free and tuned prediction generation
  training/                  # SFT / GA / DPO / NPO and training loop
  evaluation/                # RSS / CDP / JDR and harmonic mean
  experiments/               # Runtime train/evaluate/adaptation/utility implementations
  _vendor/                   # Only the required lmms-eval subset + its license
  cli.py, report.py, score.py
  utils/
docs/                        # Standalone project page and paper figures
```

## Citation and licenses

```bibtex
@inproceedings{caldarella2027who,
  title     = {Who is in the Image? Benchmarking Any Identity Anonymization in Multimodal Large Language Models},
  author    = {Caldarella, Simone and Farina, Matteo and Mancini, Massimiliano and Ricci, Elisa and Aljundi, Rahaf},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for retained upstream code licenses. An overall license for the research code and dataset redistribution should be specified by the authors before public release; this cleanup does not assign new rights to third-party imagery or model weights.
