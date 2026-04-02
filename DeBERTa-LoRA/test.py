import os
import json
import random
import logging
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(output_dir: str, filename: str = "test.log"):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("deberta_lora_test")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(os.path.join(output_dir, filename), mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


class CSVStegoDataset(Dataset):
    def __init__(self, csv_file, split_name="dataset", max_samples=None):
        df = pd.read_csv(csv_file)

        text_col = next(
            (col for col in df.columns if col.lower() in ["text", "sentence", "data"]),
            df.columns[0],
        )
        label_col = next(
            (col for col in df.columns if col.lower() in ["label", "target"]),
            df.columns[1],
        )

        df = df.dropna(subset=[text_col, label_col])

        if max_samples is not None and len(df) > max_samples:
            half = max_samples // 2
            df = (
                df.groupby(label_col, group_keys=False)
                .apply(lambda x: x.sample(n=min(len(x), half), random_state=42))
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
            )

        self.texts = df[text_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()

        num_cover = sum(1 for x in self.labels if x == 0)
        num_stego = sum(1 for x in self.labels if x == 1)
        print(
            f"[{split_name}] Loaded {len(self.texts)} samples "
            f"(cover={num_cover}, stego={num_stego})"
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    return {
        "texts": [item["text"] for item in batch],
        "labels": torch.stack([item["label"] for item in batch]),
    }


class DebertaLoRAClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.4,
    ):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query_proj", "key_proj", "value_proj", "dense_proj"],
        )
        self.model = get_peft_model(self.model, lora_config)

    def forward(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.logits


def evaluate(model, data_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing", leave=False):
            texts = batch["texts"]
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(texts)
                loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(len(data_loader), 1)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=["Cover (0)", "Stego (1)"],
        digits=4,
    )

    metrics = {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics, report


def main():
    parser = argparse.ArgumentParser(description="Test DeBERTa-LoRA for linguistic steganalysis")
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument("--domain", type=str, choices=["Movie", "News", "Twitter"], required=True)
    parser.add_argument("--algorithm", type=str, choices=["AC", "DI", "VS"], required=True)

    parser.add_argument("--split", type=str, choices=["train", "dev", "test"], default="test")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.4)

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    output_dir = Path(args.output_dir) / f"test_{args.algorithm}_{args.domain}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(str(output_dir), filename="test.log")

    logger.info("===== DeBERTa-LoRA Testing =====")
    logger.info(json.dumps(vars(args), indent=2))

    data_file = Path(args.data_root) / args.domain / args.algorithm / f"{args.split}.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing data file: {data_file}")

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    dataset = CSVStegoDataset(
        str(data_file),
        split_name=f"{args.split}-{args.domain}-{args.algorithm}",
        max_samples=args.max_samples,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = DebertaLoRAClassifier(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    metrics, report = evaluate(model, data_loader, device)

    logger.info(
        f"[{args.domain}-{args.algorithm}-{args.split}] "
        f"loss={metrics['loss']:.4f}, "
        f"acc={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}"
    )
    logger.info("\n" + report)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("Testing completed.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
