import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
import time
import numpy as np
import subprocess
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import warnings

# Use pathlib for robust path handling across OS
from pathlib import Path

warnings.filterwarnings('ignore')

# --- Auto-select Best GPU ---
def set_best_gpu():
    """
    Automatically selects the GPU with the most free memory using nvidia-smi.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        return "cpu"

    try:
        # Run nvidia-smi to query memory usage
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            capture_output=True, text=True, check=True
        )
        free_memory = [int(x) for x in result.stdout.strip().split('\n')]
        
        # Select GPU with max free memory
        best_gpu_id = np.argmax(free_memory)
        print(f"GPUs detected. Auto-selecting GPU {best_gpu_id} with {free_memory[best_gpu_id]} MiB free memory.")
        
        # Set environment variable so PyTorch only sees this GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_id)
        return f"cuda:{best_gpu_id}"
        
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Could not run nvidia-smi: {e}. Defaulting to GPU 0.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        return "cuda:0"


# --- Logger Setup ---
def get_logger(logger_name, output_dir, filename="training_log.txt"):
    """Configures and returns a logger."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, filename), mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# --- Dataset Definition ---
class StegoTextDataset(Dataset):
    """Dataset for Steganographic Text Classification."""
    def __init__(self, cover_file, stego_file):
        with open(cover_file, 'r', encoding='utf-8') as f:
            self.cover_texts = [line.strip() for line in f if line.strip()]
        with open(stego_file, 'r', encoding='utf-8') as f:
            self.stego_texts = [line.strip() for line in f if line.strip()]

        self.texts = self.cover_texts + self.stego_texts
        self.labels = [0] * len(self.cover_texts) + [1] * len(self.stego_texts)
        print(f"Dataset Stats: {len(self.cover_texts)} Normal (Label 0), {len(self.stego_texts)} Stego (Label 1)")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'label': torch.tensor(self.labels[idx], dtype=torch.long)}

# --- Model Definition ---
class DebertaClassifier(nn.Module):
    """Sequence Classification Model using DeBERTa with LoRA."""
    def __init__(self, model_name):
        super().__init__()
        # Load pre-trained DeBERTa for sequence classification
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Configure LoRA for Parameter-Efficient Fine-Tuning
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=64,
            lora_alpha=128,
            target_modules=["query_proj", "key_proj", "value_proj", "dense_proj"],
            lora_dropout=0.1,
        )
        self.bert = get_peft_model(self.bert, lora_config)
        print("LoRA adapters added to the DeBERTa model.")
        self.bert.print_trainable_parameters()

    def forward(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.bert.device)
        outputs = self.bert(**inputs)
        return outputs.logits

# --- Collate Function ---
def collate_fn(batch):
    """Collates batch data into tensors."""
    return {
        'texts': [item['text'] for item in batch],
        'labels': torch.stack([item['label'] for item in batch])
    }

# --- Training Loop ---
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, model_save_path, logger, device):
    """Executes the training and validation loop."""
    model.to(device)
    
    start_time = time.time() 
    max_memory_allocated = 0.0 # GB
    if 'cuda' in str(device):
        torch.cuda.reset_peak_memory_stats(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=3,
        num_training_steps=len(train_loader) * num_epochs
    )
    
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        
        for batch in progress_bar:
            texts, labels = batch['texts'], batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Track peak memory usage
        if 'cuda' in str(device):
            current_peak_memory = torch.cuda.max_memory_allocated(device) / (1024**3) 
            max_memory_allocated = max(max_memory_allocated, current_peak_memory)
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
                texts, labels = batch['texts'], batch['labels'].to(device)
                outputs = model(texts)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels).item()
                total_samples += len(labels)
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_samples
        
        log_message = (
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )
        logger.info(log_message)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            # Save only LoRA parameters
            torch.save(model.state_dict(), model_save_path) 
            logger.info(f"New best model saved with accuracy: {best_accuracy:.4f}")

    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info("====================================")
    logger.info("--- Total Training Cost Summary ---")
    logger.info(f"Total Training Time: {total_time:.2f} seconds")
    logger.info(f"Max GPU Memory Allocated (Peak): {max_memory_allocated:.4f} GB")
    logger.info(f"Total Trainable Parameters (LoRA): {model.bert.num_parameters(only_trainable=True)/1e6:.2f} Million")
    logger.info("====================================")


# --- Main Function ---
def main():
    device = set_best_gpu()
    
    # --- Path Configuration (Relative Paths) ---
    # Define root relative to this script
    ROOT_DIR = Path(__file__).resolve().parent
    DATA_DIR = ROOT_DIR / "data"
    OUTPUT_DIR = ROOT_DIR / "outputs"
    
    # Ensure dataset exists before running
    # Example: ./data/di/news/cover.txt
    cover_path = DATA_DIR / "di" / "news" / "cover.txt"
    stego_path = DATA_DIR / "di" / "news" / "stego.txt"
    
    if not cover_path.exists() or not stego_path.exists():
        raise FileNotFoundError(f"Data files not found at {cover_path} or {stego_path}. Please check the 'data' directory.")

    CONFIG = {
        # Automatically download from Hugging Face Hub
        "model_name": "microsoft/deberta-v3-large",
        
        "cover_file": str(cover_path),
        "stego_file": str(stego_path),
        
        "log_dir": str(OUTPUT_DIR / "logs"),
        "log_filename": "training_log.txt",
        
        # Save checkpoints inside the project folder
        "model_save_path": str(OUTPUT_DIR / "checkpoints" / "best_model.pth"),
        
        "batch_size": 12,
        "num_epochs": 10,
        "learning_rate": 5e-5,
        "train_val_split_ratio": 0.9,
    }
    
    logger = get_logger("debert_training", CONFIG["log_dir"], CONFIG["log_filename"])
    logger.info("--- Experiment Start ---")
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration: {CONFIG}")
    
    # Prepare Dataset
    dataset = StegoTextDataset(CONFIG["cover_file"], CONFIG["stego_file"])
    train_size = int(CONFIG["train_val_split_ratio"] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)
    
    # Initialize Model
    logger.info(f"Initializing model: {CONFIG['model_name']}")
    model = DebertaClassifier(model_name=CONFIG["model_name"])
    
    # Start Training
    logger.info("Starting training...")
    train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        num_epochs=CONFIG["num_epochs"], 
        learning_rate=CONFIG["learning_rate"],
        model_save_path=CONFIG["model_save_path"],
        logger=logger,
        device=torch.device(device) 
    )
    logger.info("--- Experiment End ---")

if __name__ == "__main__":
    main()