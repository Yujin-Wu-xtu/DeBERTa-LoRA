<div align="center">

# Task-Aligned Discriminative Encoders<br>for Efficient Linguistic Steganalysis

**Yanchun Li, Yujin Wu, Ming Yang, Hangtao Zhang, Dongsu Shen, and Li Zeng**

*Task-model alignment for accurate and efficient cover/stego detection.*

![ICASSP 2027 submission](https://img.shields.io/badge/ICASSP_2027-Submitted-2563eb?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-Implementation-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![DeBERTa-v3 and LoRA](https://img.shields.io/badge/DeBERTa--v3-LoRA-8b5cf6?style=flat-square)
![Datasets](https://img.shields.io/badge/Datasets-3_Domains_%C3%97_3_Algorithms-059669?style=flat-square)

[📖 Overview](#overview) · [🧩 Method](#method) · [🏆 Results](#results) · [🗂️ Datasets](#datasets) · [🚀 Quick Start](#installation) · [📚 Citation](#citation)

</div>

---

> **📬 ICASSP 2027 submission**<br>
> This repository provides the PyTorch implementation and processed datasets accompanying the submitted manuscript. The implementation retains the name **DeBERTa-LoRA**.

<a id="overview"></a>

## 📖 Overview

Does reliable linguistic steganalysis inherently require large-scale language models? We revisit this question from a **task-model alignment** perspective. Secret-message constraints can introduce weak deviations in token distributions and contextual dependencies, even when generated text remains fluent. Discriminatively pretrained encoders provide representation priors that may be well suited to detecting these traces.

We instantiate this perspective with **DeBERTa-v3**, freeze its pretrained backbone, and train **LoRA adapters and a lightweight sequence classifier**. Experiments on three text domains and three steganography algorithms show strong detection and generalization with substantially lower computational cost than a representative 7B LLM detector.

---

<a id="method"></a>

## 🧩 Method at a Glance

<div align="center">

**📝 Input Text → 🔍 Frozen DeBERTa-v3 + LoRA → 🎯 Cover / Stego**

*Discriminative pretraining · Content–position modeling · Lightweight adaptation*

</div>

### 🔍 Discriminative Representations

DeBERTa-v3's Replaced Token Detection (RTD) pretraining learns to distinguish original tokens from contextually plausible replacements. This motivates its use as a representation prior for weak distributional deviations. RTD is not re-optimized during downstream steganalysis.

### 🧭 Content–Position Modeling

Disentangled Attention (DA) separately represents token content and relative position, providing a complementary inductive bias for contextual deviations introduced by constrained generation. RTD and DA are existing pretrained properties, not new steganalysis-specific modules.

### 🪶 Parameter-Efficient Adaptation

The pretrained backbone remains frozen while low-rank updates and the classification head are optimized for cover/stego prediction. LoRA constrains the adaptation space while retaining the pretrained representation prior.

The paper's central contribution is the investigation of task-model alignment for efficient linguistic steganalysis. Supporting ablations examine the relevance of discriminative pretraining and content-position modeling, while comparisons with full fine-tuning assess the effectiveness of LoRA adaptation.

---

<a id="results"></a>

## 🏆 Main Results

> 🎯 **89.90% average ACC** · First in **8 / 9** in-domain settings<br>
> 🌍 **+7.93 points** cross-domain ACC · **+14.25 points** cross-steganography ACC<br>
> ⚡ **2.4× faster training** · **74% less peak GPU memory** at matched batch size 4<br>
> Gains and efficiency comparisons are relative to **GS-Llama7b**.

ACC and F1 values below are percentages. Improvements are absolute percentage points. These are the manuscript's reported results, not measurements from a new run of this repository.

### 🎯 In-Domain Detection

Our detector ranks first in **8 of 9** domain–algorithm settings and second on AC–Twitter among the compared methods.

<details>
<summary><b>📊 View all nine domain–algorithm results</b></summary>

| Algorithm | Domain | GS-Llama7b ACC | GS-Llama7b F1 | Ours ACC | Ours F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| AC | Movie | 85.45 | 85.27 | 87.50 | 87.39 |
| AC | News | 88.50 | 87.67 | 91.75 | 91.75 |
| AC | Twitter | 83.75 | 83.87 | 83.15 | 82.75 |
| DI | Movie | 90.75 | 90.81 | 91.16 | 91.20 |
| DI | News | 94.75 | 94.76 | 95.83 | 95.83 |
| DI | Twitter | 85.50 | 85.50 | 90.00 | 90.13 |
| VS | Movie | 84.20 | 85.23 | 93.55 | 93.54 |
| VS | News | 87.90 | 87.78 | 90.65 | 90.76 |
| VS | Twitter | 80.25 | 80.29 | 85.55 | 86.08 |
| **Average** | | **86.78** | **86.80** | **89.90** | **89.94** |

</details>

The average gains over GS-Llama7b are **3.12 points in ACC** and **3.14 points in F1**. Table 1 of the manuscript also compares TS-RNN, TS-FCN, TS-CSW, EILGF, LSTMATT, and directly comparable published STLC-KG results.

### 🌍 Generalization under Distribution Shift

Cross-domain results average all six domain-transfer directions for each algorithm. Cross-steganography results average all six unseen-algorithm transfers on **News**.

| Evaluation | GS-Llama7b ACC | GS-Llama7b F1 | Ours ACC | Ours F1 | ACC gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross-domain: AC | 60.54 | 59.10 | 69.20 | 66.45 | +8.66 |
| Cross-domain: DI | 62.54 | 62.59 | 74.58 | 72.76 | +12.04 |
| Cross-domain: VS | 80.42 | 80.43 | 83.51 | 83.37 | +3.09 |
| **Cross-domain: overall** | **67.83** | **67.37** | **75.76** | **74.19** | **+7.93** |
| **Cross-steganography: News** | **60.63** | **60.28** | **74.88** | **74.33** | **+14.25** |

### ⚡ Computational Efficiency

Measurements reported on a single **NVIDIA RTX 4090**:

| Model | Batch size | Training time (min) ↓ | Peak GPU memory (GB) ↓ |
| --- | ---: | ---: | ---: |
| Ours | 4 | 25.35 | 3.04 |
| Ours | 32 | 4.10 | 5.05 |
| GS-Llama7b | 4 | 61.73 | 11.84 |

At the matched batch size of 4, our detector trains approximately **2.4× faster** and uses **74% less peak GPU memory** than GS-Llama7b.

---

<a id="datasets"></a>

## 🗂️ Datasets

The released data cover **🎬 Movie**, **📰 News**, and **💬 Twitter**, each with three steganography algorithms:

| Directory | Steganography algorithm |
| --- | --- |
| `AC` | Arithmetic Coding |
| `DI` | Discop |
| `VS` | VAE-Stega |

Each domain–algorithm pair contains an independent balanced dataset:

| Split | File | Total samples | Cover | Stego |
| --- | --- | ---: | ---: | ---: |
| Training | `train.csv` | 10,000 | 5,000 | 5,000 |
| Validation | `dev.csv` | 1,000 | 500 | 500 |
| Test | `test.csv` | 2,000 | 1,000 | 1,000 |

CSV files contain `sentence` and `label` columns; their order can vary. Labels are **0 = cover** and **1 = stego**. The manuscript uses steganographic samples with mixed embedding rates from **1 to 5 bits per word (bpw)**.

---

<a id="repository"></a>

## 📁 Repository Structure

```text
DeBERTa-LoRA/
├── datasets/
│   ├── Movie/
│   │   ├── AC/                 # train.csv, dev.csv, test.csv
│   │   ├── DI/                 # train.csv, dev.csv, test.csv
│   │   └── VS/                 # train.csv, dev.csv, test.csv
│   ├── News/                   # Same algorithm/split structure
│   └── Twitter/                # Same algorithm/split structure
├── train.py
├── test.py
├── environment.yaml
└── README.md
```

This repository releases our detector and processed datasets. Baseline implementations are not bundled. The paper evaluates TS-RNN, TS-FCN, TS-CSW, EILGF, LSTMATT, and GS-Llama7b using their public implementations or configurations; STLC-KG comparisons use only directly comparable published results.

---

<a id="installation"></a>

## 🚀 Quick Start

```bash
git clone https://github.com/Yujin-Wu-xtu/DeBERTa-LoRA.git
cd DeBERTa-LoRA
conda env create -f environment.yaml
conda activate stego_env
```

Run the commands below from the repository root. The scripts default to `microsoft/deberta-v3-large`; the tokenizer and pretrained model are downloaded on first use unless already cached. A CUDA-capable GPU is recommended for training.

---

<a id="training"></a>

## 🏋️ Training

Train on one domain–algorithm pair, selecting the checkpoint with the highest validation accuracy:

```bash
python train.py \
  --train_domain Movie \
  --algorithm AC \
  --model_name microsoft/deberta-v3-large \
  --batch_size 12 \
  --epochs 10 \
  --learning_rate 5e-5 \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.1 \
  --output_dir outputs
```

This saves the checkpoint to `outputs/AC_Movie/best_model.pt` and training logs to `outputs/AC_Movie/train.log`, then evaluates the selected checkpoint on the in-domain test split. Add `--cross_domain` to also evaluate it on all three domains for the same algorithm; the source-domain result is included in these logs.

<details>
<summary><b>🔧 Manuscript settings and current script defaults</b></summary>

| Setting | Submitted manuscript | Current training script default |
| --- | --- | --- |
| LoRA rank / alpha | 64 / 128 | 64 / 128 |
| LoRA dropout | 0.1 | 0.4 |
| Maximum input length | 512 tokens | 512 tokens |
| Epochs | 10 | 5 |
| Batch size | 12 | 4 |
| Optimizer / learning rate | AdamW / 5e-5 | AdamW / 5e-5 |
| Scheduler | Linear, 3 warm-up steps | Linear, 10% of total training steps as warm-up |
| Checkpoint selection | Highest validation ACC | Highest validation ACC |

The command above explicitly sets the manuscript's exposed training parameters. **The released script still uses a 10% warm-up schedule and has no command-line option for the manuscript's 3 warm-up steps**, so the command alone does not reproduce every reported setting. The released classifier uses two output logits with cross-entropy; the manuscript formulates binary classification with a sigmoid and binary cross-entropy.

</details>

### 🔥 Quick Smoke Run

For a small smoke run using a separate output directory:

```bash
python train.py \
  --train_domain Movie \
  --algorithm AC \
  --epochs 1 \
  --batch_size 4 \
  --lora_dropout 0.1 \
  --max_train_samples 200 \
  --max_dev_samples 100 \
  --max_test_samples 100 \
  --output_dir outputs/smoke
```

---

<a id="evaluation"></a>

## 🧪 Evaluation

### 🎯 In-Domain Evaluation

Evaluate the checkpoint trained above:

```bash
python test.py \
  --domain Movie \
  --algorithm AC \
  --checkpoint outputs/AC_Movie/best_model.pt \
  --lora_dropout 0.1
```

Use the same model name, LoRA rank, and LoRA alpha as during training. `--split` defaults to `test`; use `--split dev` for validation data. Metrics, a classification report, and logs are saved as `metrics.json`, `classification_report.txt`, and `test.log` under `outputs/test_AC_Movie/`.

### 🌍 Cross-Domain Evaluation

For cross-domain evaluation, select the target domain while retaining the source checkpoint. For example, Movie → News on AC:

```bash
python test.py \
  --domain News \
  --algorithm AC \
  --checkpoint outputs/AC_Movie/best_model.pt \
  --lora_dropout 0.1 \
  --output_dir outputs/cross_domain/Movie_to_News
```

### 🔄 Cross-Steganography Evaluation

For cross-steganography evaluation on News, first train an AC–News checkpoint using the training command with `--train_domain News`, then evaluate it on DI–News:

```bash
python test.py \
  --domain News \
  --algorithm DI \
  --checkpoint outputs/AC_News/best_model.pt \
  --lora_dropout 0.1 \
  --output_dir outputs/cross_stego/AC_to_DI
```

Use a separate `--output_dir` for each transfer experiment to retain its evaluation files.

---

<a id="citation"></a>

## 📚 Citation

The manuscript has been **submitted to ICASSP 2027**. Until publication details are available, please use this provisional manuscript citation:

```bibtex
@unpublished{li2026taskaligned,
  title  = {Task-Aligned Discriminative Encoders for Efficient Linguistic Steganalysis},
  author = {Li, Yanchun and Wu, Yujin and Yang, Ming and Zhang, Hangtao and Shen, Dongsu and Zeng, Li},
  year   = {2026},
  note   = {Manuscript submitted to ICASSP 2027},
  url    = {https://github.com/Yujin-Wu-xtu/DeBERTa-LoRA}
}
```
