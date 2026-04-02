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
- the **experimental datasets** used in our study

Recent advances in Large Language Models (LLMs) have significantly improved the quality of generative linguistic steganography, making hidden information more natural and difficult to detect. To address this challenge, we propose **DeBERTa-LoRA**, an efficient yet effective framework that combines the **DeBERTa-v3** backbone with **Low-Rank Adaptation (LoRA)** for high-performance linguistic steganalysis.

Unlike LLM-based detection approaches that rely on large-scale generative models and expensive fine-tuning, our method emphasizes **architectural alignment** between the model and steganographic artifacts.

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
- **~15× faster training**
- **~70% lower peak GPU memory**

Our experiments cover multiple steganographic paradigms, including:

- **AC** (Arithmetic Coding-based steganography)
- **DI** (Discop-based steganography)
- **VS** (VAE-Stega-based steganography)

and multiple text domains:

- **Movie**
- **News**
- **Twitter**

---

## 📂 What This Repository Contains

This repository is intended to release:

- our **proposed method**
- our **processed experimental datasets**

It does **not** include re-implementations of all baseline methods.

For baseline comparisons in the paper, we mainly used the **official open-source repositories** released by the original authors whenever available. Please refer to the corresponding papers or official repositories for those implementations.

---

## 📁 Project Structure

```text
DeBERTa-LoRA/
├── datasets/                   # Released datasets
│   ├── AC/
│   │   ├── Movie/
│   │   ├── News/
│   │   └── Twitter/
│   ├── DI/
│   │   ├── Movie/
│   │   ├── News/
│   │   └── Twitter/
│   └── VS/
│       ├── Movie/
│       ├── News/
│       └── Twitter/
│
├── src/                        # Source code
│   ├── train.py                # Training script
│   ├── test.py                 # Evaluation script
│   ├── model.py                # Model definition
│   ├── dataset.py              # Data loading utilities
│   └── utils.py                # Helper functions
│
├── checkpoints/                # Saved checkpoints (optional)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
🗂 Dataset Format

Each dataset directory contains paired cover and stego texts.

Example:

datasets/AC/News/
├── train_cover.txt
├── train_stego.txt
├── test_cover.txt
└── test_stego.txt

Each line in a file corresponds to one text sample.

If your released version uses another file organization format, please adjust this section accordingly.

⚙️ Installation

We recommend using Python 3.8+ with a clean virtual environment.

pip install -r requirements.txt
🏋️ Training

Run the training script:

python src/train.py

You may modify key hyperparameters such as:

model name
batch size
learning rate
number of epochs
dataset path
LoRA rank / dropout

depending on your experimental setting.

🧪 Evaluation

To evaluate a trained checkpoint:

python src/test.py

Evaluation metrics typically include:

Accuracy
Precision
Recall
F1-score
📌 Notes on Baselines

This repository focuses on our method and our released datasets.

Baseline models compared in the paper are not bundled here, because most of them already have public implementations released by their original authors. This helps keep the repository lightweight, clean, and easier to maintain.

📚 Citation

If you find this repository useful, please cite our paper:

@article{debloRA_steganalysis,
  title={Efficient Yet Effective: DeBERTa-LoRA for Linguistic Steganalysis},
  author={...},
  journal={IEEE Signal Processing Letters},
  year={2026}
}

Please replace the citation information with the final published version after acceptance.
