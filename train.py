import os
import time
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
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score

warnings.filterwarnings("ignore")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(output_dir: str, filename: str = "train.log"):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("deberta_lora")
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
        self.model.print_trainable_parameters()

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


def evaluate(model, data_loader, device, use_amp=True):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    autocast_enabled = use_amp and device.type == "cuda"

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            texts = batch["texts"]
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                logits = model(texts)
                loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            total_loss += loss.item()
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(len(data_loader), 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=["Cover (0)", "Stego (1)"],
        digits=4,
    )
    return avg_loss, acc, f1, report


def train_one_run(args):
    set_seed(args.seed)
    device = get_device()

    output_dir = Path(args.output_dir) / f"{args.algorithm}_{args.train_domain}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(str(output_dir), "train.log")

    logger.info("===== DeBERTa-LoRA Training =====")
    logger.info(json.dumps(vars(args), indent=2))

    train_file = Path(args.data_root) / args.train_domain / args.algorithm / "train.csv"
    dev_file = Path(args.data_root) / args.train_domain / args.algorithm / "dev.csv"
    test_file = Path(args.data_root) / args.train_domain / args.algorithm / "test.csv"

    if not train_file.exists():
        raise FileNotFoundError(f"Missing train file: {train_file}")
    if not dev_file.exists():
        raise FileNotFoundError(f"Missing dev file: {dev_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Missing test file: {test_file}")

    train_dataset = CSVStegoDataset(
        str(train_file),
        split_name=f"train-{args.train_domain}-{args.algorithm}",
        max_samples=args.max_train_samples,
    )
    dev_dataset = CSVStegoDataset(
        str(dev_file),
        split_name=f"dev-{args.train_domain}-{args.algorithm}",
        max_samples=args.max_dev_samples,
    )
    test_dataset = CSVStegoDataset(
        str(test_file),
        split_name=f"test-{args.train_domain}-{args.algorithm}",
        max_samples=args.max_test_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
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
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    total_steps = max((len(train_loader) // max(args.grad_accum_steps, 1)) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_dev_acc = 0.0
    best_ckpt = output_dir / "best_model.pt"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        total_train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress_bar):
            texts = batch["texts"]
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(texts)
                loss = criterion(logits, labels)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            total_train_loss += loss.item() * args.grad_accum_steps

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            progress_bar.set_postfix(loss=f"{loss.item() * args.grad_accum_steps:.4f}")

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        dev_loss, dev_acc, dev_f1, _ = evaluate(model, dev_loader, device)

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"dev_loss={dev_loss:.4f} | "
            f"dev_acc={dev_acc:.4f} | "
            f"dev_f1={dev_f1:.4f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), best_ckpt)
            logger.info(f"Saved new best checkpoint to {best_ckpt}")

    total_time = time.time() - start_time
    peak_mem_gb = 0.0
    if device.type == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    logger.info(f"Training finished in {total_time:.2f} s")
    logger.info(f"Peak GPU memory: {peak_mem_gb:.4f} GB")

    logger.info("Loading best model for in-domain test...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_loss, test_acc, test_f1, test_report = evaluate(model, test_loader, device)
    logger.info(
        f"[In-domain Test] loss={test_loss:.4f}, acc={test_acc:.4f}, f1={test_f1:.4f}"
    )
    logger.info("\n" + test_report)

    if args.cross_domain:
        logger.info("===== Cross-domain evaluation =====")
        domain_list = ["Movie", "News", "Twitter"]
        for target_domain in domain_list:
            target_test = Path(args.data_root) / target_domain / args.algorithm / "test.csv"
            if not target_test.exists():
                logger.warning(f"Skip missing file: {target_test}")
                continue

            target_dataset = CSVStegoDataset(
                str(target_test),
                split_name=f"cross-{args.train_domain}-to-{target_domain}",
                max_samples=args.max_test_samples,
            )
            target_loader = DataLoader(
                target_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
            )
            loss, acc, f1, report = evaluate(model, target_loader, device)
            logger.info(
                f"[Cross-domain] {args.train_domain} -> {target_domain} | "
                f"loss={loss:.4f}, acc={acc:.4f}, f1={f1:.4f}"
            )
            logger.info("\n" + report)


def build_parser():
    parser = argparse.ArgumentParser(description="Train DeBERTa-LoRA for linguistic steganalysis")
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument("--train_domain", type=str, choices=["Movie", "News", "Twitter"], required=True)
    parser.add_argument("--algorithm", type=str, choices=["AC", "DI", "VS"], required=True)

    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--output_dir", type=str, default="outputs")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.4)

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_dev_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)

    parser.add_argument("--cross_domain", action="store_true")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train_one_run(args)
