import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from tqdm import tqdm
import warnings
from sklearn.metrics import classification_report
from pathlib import Path

# --- Basic Settings ---
# Note: In a real-world scenario, you might want to use argparse or the set_best_gpu function here too.
# For simplicity, we default to cuda:0 if available.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings('ignore')


# ==============================================================
#  Dataset Definition
# ==============================================================
class StegoTextDataset(Dataset):
    """Dataset for Steganographic Text Classification."""
    def __init__(self, cover_file, stego_file):
        with open(cover_file, 'r', encoding='utf-8') as f:
            self.cover_texts = [line.strip() for line in f if line.strip()]
        with open(stego_file, 'r', encoding='utf-8') as f:
            self.stego_texts = [line.strip() for line in f if line.strip()]
        self.texts = self.cover_texts + self.stego_texts
        self.labels = [0] * len(self.cover_texts) + [1] * len(self.stego_texts)
        print(f"Test Set Stats: {len(self.cover_texts)} Normal (Label 0), {len(self.stego_texts)} Stego (Label 1)")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'label': self.labels[idx]}


# ==============================================================
#  Model Definition 
# ==============================================================
class DebertaClassifier(nn.Module):
    """Sequence Classification Model using DeBERTa."""
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=64,
            lora_alpha=128,
            target_modules=["query_proj", "key_proj", "value_proj", "dense_proj"],
            lora_dropout=0.1,
        )
        self.bert = get_peft_model(self.bert, lora_config)

    def forward(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.bert.device)
        outputs = self.bert(**inputs)
        return outputs.logits


# ==============================================================
#  Collate Function
# ==============================================================
def collate_fn(batch):
    return {
        'texts': [item['text'] for item in batch],
        'labels': torch.tensor([item['label'] for item in batch], dtype=torch.long)
    }


# ==============================================================
#  Evaluation Function
# ==============================================================
def val(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            texts = batch['texts']
            labels = batch['labels'].to(device)

            logits = model(texts)
            predictions = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    print("\n" + "="*50)
    print("                   Classification Report")
    print("="*50)

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=['Normal Text (0)', 'Stego Text (1)'],
        digits=4
    )
    print(report)
    print("="*50)


# ==============================================================
#  Main Function
# ==============================================================
def main():
    # --- Path Configuration (Relative Paths) ---
    ROOT_DIR = Path(__file__).resolve().parent
    DATA_DIR = ROOT_DIR / "data"
    OUTPUT_DIR = ROOT_DIR / "outputs"

    print("Preparing test dataset...")
    
    
    cover_path = DATA_DIR / "ac" / "news" / "tc.txt"
    stego_path = DATA_DIR / "ac" / "news" / "ts.txt"
    
    if not cover_path.exists() or not stego_path.exists():
        print(f"Error: Data files not found at {cover_path}")
        return

    dataset = StegoTextDataset(str(cover_path), str(stego_path))
    val_loader = DataLoader(dataset, batch_size=12, shuffle=False, collate_fn=collate_fn)

    print("Initializing model structure...")
    # Use HuggingFace model ID for portability
    model_name = "microsoft/deberta-v3-large"
    model = DebertaClassifier(model_name=model_name)

    # Path to the trained checkpoint
    # Assumes you placed the downloaded weight in outputs/checkpoints/
    model_path = OUTPUT_DIR / "checkpoints" / "best_model.pth"
    
    print(f"Loading trained weights from '{model_path}'...")

    try:
        # Automatically map to available device
        model.load_state_dict(torch.load(str(model_path), map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at '{model_path}'!")
        print("Please ensure you have trained the model or downloaded the weights.")
        return
    except RuntimeError as e:
        print("\n" + "*"*20 + " Model Loading Error " + "*"*20)
        print("Error Message:", e)
        print("This usually means the model structure in test script mismatches the saved .pth file.")
        print("Ensure 'num_labels' and LoRA config match exactly.")
        print("*"*55)
        return

    # Start Evaluation
    val(model, val_loader)


if __name__ == "__main__":
    main()