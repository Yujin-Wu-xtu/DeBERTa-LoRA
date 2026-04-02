<div align="center">

# Efficient Yet Effective: DeBERTa-LoRA for Linguistic Steganalysis

**Official PyTorch Implementation and Dataset Release**

![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![Task](https://img.shields.io/badge/Task-Linguistic_Steganalysis-blue)
![Model](https://img.shields.io/badge/Model-DeBERTa--LoRA-orange)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📖 Introduction

This repository provides:

- the **official implementation** of **DeBERTa-LoRA** for linguistic steganalysis
- the **processed experimental datasets** used in our study

Recent advances in Large Language Models (LLMs) have greatly improved the quality of generative linguistic steganography, making hidden information more natural and harder to detect. To address this challenge, we propose **DeBERTa-LoRA**, an efficient yet effective framework that combines **DeBERTa-v3** with **Low-Rank Adaptation (LoRA)** for high-performance linguistic steganalysis.

Instead of relying on expensive large-scale generative models for detection, our method emphasizes **architectural alignment** between the backbone model and steganographic artifacts.

---

## ✨ Method Highlights

### 1. Replaced Token Detection (RTD)
RTD is naturally sensitive to the **distributional artifacts** introduced by steganographic sampling strategies, such as truncation and constrained token selection.

### 2. Disentangled Attention (DA)
DA helps capture **structural content-position misalignments** caused by message embedding constraints, enabling the model to detect subtle contextual inconsistencies.

### 3. LoRA as a Differential Feature Amplifier
Rather than using LoRA only for efficient fine-tuning, we interpret it as a **Differential Feature Amplifier (DFA)** that enhances weak steganographic signals over a frozen semantic backbone.

Together, these components make DeBERTa-LoRA both **accurate** and **computationally efficient** for practical deployment.

---

## 🚀 Main Results

Compared with leading LLM-based baselines, DeBERTa-LoRA achieves:

- **State-of-the-art detection performance**
- **Substantially faster training**
- **Much lower GPU memory consumption**

Our experiments cover:

### Text domains
- **Movie**
- **News**
- **Twitter**

### Steganographic algorithms
- **AC**
- **DI**
- **VS**

---

## 📂 Repository Scope

This repository is intended to release:

- our **proposed method**
- our **processed datasets**

It does **not** include re-implementations of all baseline methods.

For baseline comparisons in the paper, we mainly used the **official open-source implementations** released by the original authors whenever available. This keeps the repository lightweight and easier to maintain.

---

## 📁 Repository Structure

```text
DeBERTa-LoRA/
├── datasets/
│   ├── Movie/
│   │   ├── AC/
│   │   │   ├── train.csv
│   │   │   ├── dev.csv
│   │   │   └── test.csv
│   │   ├── DI/
│   │   │   ├── train.csv
│   │   │   ├── dev.csv
│   │   │   └── test.csv
│   │   └── VS/
│   │       ├── train.csv
│   │       ├── dev.csv
│   │       └── test.csv
│   ├── News/
│   └── Twitter/
│
├── train.py
├── test.py
├── requirements.txt
├── environment.yaml
├── .gitignore
└── README.md
📁 Dataset Description

We release the final processed datasets used in our experiments.

Domains
Movie
News
Twitter
Algorithms
AC
DI
VS

Each domain-algorithm pair contains three splits:

train.csv
dev.csv
test.csv

Each CSV file contains text samples and binary labels:

0 → cover text
1 → stego text
⚙️ Installation

We recommend using Python 3.8+ and a clean virtual environment.

pip install -r requirements.txt

If you prefer Conda, you may also use the provided environment.yaml.

🏋️ Training

Train a model on one domain and one algorithm:

python train.py --train_domain Movie --algorithm AC

Example:

python train.py --train_domain News --algorithm VS

To enable cross-domain evaluation after training:

python train.py --train_domain Twitter --algorithm DI --cross_domain
Optional arguments
python train.py \
  --train_domain Movie \
  --algorithm AC \
  --model_name microsoft/deberta-v3-large \
  --batch_size 4 \
  --epochs 5 \
  --learning_rate 5e-5 \
  --output_dir outputs
Quick smoke test
python train.py \
  --train_domain Movie \
  --algorithm AC \
  --max_train_samples 200 \
  --max_dev_samples 100 \
  --max_test_samples 100
🧪 Evaluation

Evaluate a saved checkpoint:

python test.py \
  --domain Movie \
  --algorithm AC \
  --checkpoint outputs/AC_Movie/best_model.pt

Evaluate on another split:

python test.py \
  --domain News \
  --algorithm VS \
  --split dev \
  --checkpoint outputs/VS_News/best_model.pt
Test outputs

The evaluation script saves:

metrics.json
classification_report.txt
test.log

under the corresponding output directory.

📌 Notes
This repository focuses on our method and our datasets.
Baseline implementations are not bundled here.
For reproducibility, all data paths in the released scripts are relative paths, making the code directly runnable after cloning the repository.
📚 Citation

If you find this repository useful, please cite our paper:

@article{deberta_lora_steganalysis,
  title={Efficient Yet Effective: DeBERTa-LoRA for Linguistic Steganalysis},
  author={...},
  journal={IEEE Signal Processing Letters},
  year={2026}
}

Please replace the citation with the final published version after acceptance.
