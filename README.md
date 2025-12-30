# Efficient Yet Effective: DeBERTa-LoRA for Linguistic Steganalysis

This repository contains the official PyTorch implementation for the paper **"Efficient Yet Effective: DeBERTa-LoRA for Linguistic Steganalysis"**, submitted to **ICME 2026**.

## 🚀 Overview

In this work, we propose an efficient steganalysis framework that synergizes the **DeBERTa-v3** architecture with **Low-Rank Adaptation (LoRA)**. By prioritizing architectural alignment over model scaling, we achieve state-of-the-art detection performance while significantly reducing computational costs.

### Key Features:
- **Architectural Alignment:** Leverages DeBERTa's *Replaced Token Detection (RTD)* to detect distributional sampling artifacts and *Disentangled Attention (DA)* for structural misalignments.
- **Differential Feature Amplifier (DFA):** We theoretically model LoRA as a high-pass filter that amplifies subtle steganographic signals over a frozen semantic backbone.
- **High Efficiency:** Compared to LLM-based methods (e.g., GS-Llama7b), our method reduces training time by **~7x** and GPU memory usage by **50%**, making it feasible for single-GPU deployment (e.g., RTX 4090).

## 📊 Performance

Our model achieves State-of-the-Art (SOTA) performance across multiple steganographic algorithms (**AC, Discop, VAE-Stega**) and domains (**Movie, News, Twitter**).

| Method | Avg. Accuracy | Param Efficiency |
| :--- | :---: | :---: |
| TS-RNN | 76.5% | - |
| GS-Llama7b | 90.5% | Low |
| **DeBERTa-LoRA (Ours)** | **94.8%** | **High** |

## 🛠️ Requirements

The code is built with Python 3.8+ and PyTorch. Key dependencies include:
* `torch>=2.0.0`
* `transformers>=4.30.0`
* `peft>=0.4.0`
* `nltk`
* `datasets`

*(Detailed `requirements.txt` will be released soon.)*

## 📂 Project Structure

```text
.
├── data/               # Dataset preprocessing scripts
├── models/             # DeBERTa-v3 and LoRA configurations
├── utils/              # Helper functions and metrics
├── train.py            # Main training script
├── evaluate.py         # Evaluation and testing script
└── README.md
