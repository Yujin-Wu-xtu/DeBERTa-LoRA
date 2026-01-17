<div align="center">

# Efficient Yet Effective: DeBERTa-LoRA for Linguistic Steganalysis

**Official PyTorch Implementation** submitted to **ICME 2026**

![Status](https://img.shields.io/badge/Status-Anonymous_Submission-orange)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

> [!IMPORTANT]
> **🔒 Anonymous Submission**
>
> Identifying information has been removed for double-blind review. This repository contains the code for the paper *"Efficient Yet Effective: DeBERTa-LoRA for Linguistic Steganalysis"*.

## 📖 Overview

The rapid development of **Large Language Models (LLMs)** has transformed generative linguistic steganography into a critical cybersecurity threat. To address this, we propose **DeBERTa-LoRA**, a framework that synergizes the DeBERTa-v3 architecture with Low-Rank Adaptation (LoRA).

Unlike methods relying on massive model scaling, our approach prioritizes **architectural alignment**:

* **Replaced Token Detection (RTD)**
    * Intrinsically discriminates distributional sampling artifacts ($\phi_{dist}$) introduced by truncation strategies.
* **Disentangled Attention (DA)**
    * Effectively captures structural content-position misalignments ($\phi_{struct}$) caused by embedding constraints.
* **Differential Feature Amplifier (DFA)**
    * We theoretically model LoRA as a **high-pass filter** that explicitly isolates and amplifies subtle steganographic signals over a frozen semantic backbone.

### 🚀 Key Results
Our method achieves **State-of-the-Art (SOTA)** performance while reducing:
* **Training Time:** ~7x faster
* **Peak GPU Memory:** 50% lower
*(Compared to leading LLM-based baselines, e.g., GS-Llama7b)*

---

## 📂 Project Structure

```text
DeBERTa-LoRA/
├── data/                 # Dataset directory
│   ├── ac/               # Example: data/ac/news/cover.txt
│   ├── di/               # Discop algorithm datasets
│   └── vs/               # VAE-Stega algorithm datasets
├── src/                  # Source code
│   ├── train.py          # Main training script (DeBERTa + LoRA)
│   └── test.py           # Evaluation script
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

⚡ Quick Start
1. Installation
We recommend using a Conda environment with Python 3.8+.

Bash

pip install -r requirements.txt
2. Data Preparation
Ensure your dataset is organized in the data/ directory. Each file should contain one text sample per line.

Directory Layout: data/{algorithm}/{domain}/

Required Files: cover.txt and stego.txt

3. Training
Run the training script to fine-tune the model. The script automatically selects the best available GPU.

Bash

python src/train.py
Configuration: You can adjust hyperparameters (Batch Size, LR, Epochs) and dataset paths directly in the CONFIG dictionary within src/train.py. The model defaults to microsoft/deberta-v3-large.

4. Evaluation
To evaluate the trained model using the best checkpoint:

Bash

python src/test.py
For any questions regarding the code or paper, please open an anonymous issue in this repository.
