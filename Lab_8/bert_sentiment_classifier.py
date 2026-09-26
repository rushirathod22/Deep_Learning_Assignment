"""
=============================================================================
Deep Learning Lab - Assignment 8
Title: Implementation of Pretrained BERT Model for Sentiment Analysis
Course: B.Tech Computer Science & Engineering (Artificial Intelligence)
=============================================================================
"""

import os
import sys

# Prevent transformers from loading broken tensorflow protobuf bindings
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.modules['tensorflow'] = None

import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration & Hyperparameters
# ---------------------------------------------------------------------------
CONFIG = {
    "model_name": "bert-base-uncased",
    "data_path": "data/sentiment_dataset.csv",
    "output_dir": "saved_model",
    "plots_dir": "evaluation_plots",
    "max_len": 128,
    "batch_size": 16,
    "epochs": 3,
    "learning_rate": 2e-5,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "random_seed": 42
}

def set_seed(seed=42):
    """Ensure reproducibility across runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------
# PyTorch Dataset Definition
# ---------------------------------------------------------------------------
class SentimentDataset(Dataset):
    """
    Custom PyTorch Dataset for tokenizing and feeding text into BERT.
    """
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item] if self.labels is not None else None

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,       # Adds [CLS] and [SEP]
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        item_dict = {
            "review_text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten()
        }

        if label is not None:
            item_dict["labels"] = torch.tensor(label, dtype=torch.long)

        return item_dict

# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------
def load_and_preprocess_data(csv_path, tokenizer, test_size=0.15, val_size=0.15, batch_size=16, max_len=128):
    """Loads CSV, performs stratified train/val/test splits, and creates DataLoaders."""
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {len(df)} samples.")
    print("Class distribution:\n", df['sentiment'].value_counts().to_dict())

    # Split into Train + Val and Test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['sentiment'],
        random_state=CONFIG['random_seed']
    )

    # Relative validation size
    val_rel_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_rel_size,
        stratify=train_val_df['sentiment'],
        random_state=CONFIG['random_seed']
    )

    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Create PyTorch datasets
    train_dataset = SentimentDataset(train_df['review_text'].values, train_df['sentiment'].values, tokenizer, max_len)
    val_dataset = SentimentDataset(val_df['review_text'].values, val_df['sentiment'].values, tokenizer, max_len)
    test_dataset = SentimentDataset(test_df['review_text'].values, test_df['sentiment'].values, tokenizer, max_len)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, test_df

# ---------------------------------------------------------------------------
# Training & Validation Functions
# ---------------------------------------------------------------------------
def train_epoch(model, data_loader, optimizer, scheduler, device):
    """Performs one training epoch."""
    model.train()
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1)
        correct_preds += torch.sum(preds == labels).item()
        total_samples += labels.size(0)
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['max_grad_norm'])
        optimizer.step()
        scheduler.step()

    epoch_loss = total_loss / len(data_loader)
    epoch_acc = correct_preds / total_samples
    return epoch_loss, epoch_acc


def evaluate(model, data_loader, device):
    """Evaluates model performance on validation or test sets."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
        "probabilities": np.array(all_probs)
    }
    return metrics

# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------
def run_training_pipeline():
    set_seed(CONFIG['random_seed'])
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs(CONFIG['plots_dir'], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"  BERT Fine-Tuning Pipeline for Sentiment Analysis")
    print(f"  Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"=======================================================\n")

    # 1. Tokenizer
    print("Loading Pretrained BERT Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])

    # 2. Dataset & DataLoaders
    train_loader, val_loader, test_loader, test_df = load_and_preprocess_data(
        CONFIG['data_path'],
        tokenizer,
        batch_size=CONFIG['batch_size'],
        max_len=CONFIG['max_len']
    )

    # 3. Model
    print(f"\nLoading Pretrained BERT Sequence Classification Model ({CONFIG['model_name']})...")
    model = BertForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(device)

    # 4. Optimizer & Scheduler
    total_steps = len(train_loader) * CONFIG['epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], eps=CONFIG['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"Total Training Batches per Epoch: {len(train_loader)}")
    print(f"Total Optimization Steps: {total_steps} (Warmup steps: {warmup_steps})")

    # Tracking metrics
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }

    best_val_f1 = 0.0
    start_time = time.time()

    print("\n----------------- Beginning Model Training -----------------")
    for epoch in range(1, CONFIG['epochs'] + 1):
        ep_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics = evaluate(model, val_loader, device)
        ep_time = time.time() - ep_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        print(f"Epoch {epoch}/{CONFIG['epochs']} [{ep_time:.1f}s]: "
              f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc*100:.2f}% | "
              f"Val Loss = {val_metrics['loss']:.4f}, Val Acc = {val_metrics['accuracy']*100:.2f}%, "
              f"Val F1 = {val_metrics['f1']:.4f}")

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            print(f"  -> Best model validation F1 improved to {best_val_f1:.4f}. Saving checkpoint...")
            model.save_pretrained(CONFIG['output_dir'])
            tokenizer.save_pretrained(CONFIG['output_dir'])

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time:.2f} seconds.")

    # -----------------------------------------------------------------------
    # Evaluation on Hold-Out Test Set
    # -----------------------------------------------------------------------
    print("\n----------------- Evaluating on Hold-Out Test Set -----------------")
    # Load best model for evaluation
    best_model = BertForSequenceClassification.from_pretrained(CONFIG['output_dir'])
    best_model.to(device)

    test_metrics = evaluate(best_model, test_loader, device)
    y_test = test_metrics["labels"]
    y_pred = test_metrics["predictions"]

    print(f"\nFinal Test Set Results:")
    print(f"Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {test_metrics['precision']*100:.2f}%")
    print(f"Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"F1 Score:  {test_metrics['f1']*100:.2f}%")

    print("\nDetailed Classification Report:")
    target_names = ["Negative (0)", "Positive (1)"]
    report_str = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print(report_str)

    # -----------------------------------------------------------------------
    # Generate & Save Visualization Plots
    # -----------------------------------------------------------------------
    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title("BERT Sentiment Analysis - Confusion Matrix", fontsize=13, fontweight='bold')
    plt.xlabel("Predicted Sentiment", fontsize=11)
    plt.ylabel("True Sentiment", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(CONFIG['plots_dir'], "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix to: {cm_path}")

    # 2. Training Curves Plot
    epochs_range = range(1, CONFIG['epochs'] + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs_range, history["train_loss"], 'o-', color='#1f77b4', label="Training Loss", lw=2)
    ax1.plot(epochs_range, history["val_loss"], 's--', color='#ff7f0e', label="Validation Loss", lw=2)
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_xticks(epochs_range)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.plot(epochs_range, [acc * 100 for acc in history["train_acc"]], 'o-', color='#2ca02c', label="Train Accuracy", lw=2)
    ax2.plot(epochs_range, [acc * 100 for acc in history["val_acc"]], 's--', color='#d62728', label="Val Accuracy", lw=2)
    ax2.set_title("Classification Accuracy (%) vs. Epochs", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_xticks(epochs_range)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.suptitle("BERT Pretrained Model Training & Validation Performance", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    curves_path = os.path.join(CONFIG['plots_dir'], "training_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"Saved Training Curves to: {curves_path}")

    # -----------------------------------------------------------------------
    # Interactive Sample Predictions
    # -----------------------------------------------------------------------
    print("\n----------------- Real-Time Sample Predictions -----------------")
    sample_texts = [
        "The cinematography was breathtaking and the storyline kept me hooked until the final scene!",
        "An utter waste of time; the script was terrible and the characters were completely unconvincing.",
        "Not bad at all! Exceeded my initial expectations and delivered solid entertainment.",
        "I wanted to love this so much, but the constant plot holes completely ruined the experience."
    ]

    for sample in sample_texts:
        res = predict_sentiment(sample, best_model, tokenizer, device)
        print(f"\nReview: \"{sample}\"")
        print(f" -> Predicted Sentiment: {res['sentiment']} (Confidence: {res['confidence']*100:.2f}%)")
        print(f" -> Softmax Probabilities: Negative={res['prob_negative']*100:.2f}%, Positive={res['prob_positive']*100:.2f}%")

    print("\n[Done] Model training, evaluation, and inference demonstration completed successfully!")

# ---------------------------------------------------------------------------
# Inference Function
# ---------------------------------------------------------------------------
def predict_sentiment(text, model, tokenizer, device=None):
    """
    Predict sentiment for a given input sentence using the fine-tuned BERT model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=CONFIG['max_len'],
        padding="max_length",
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        prediction = np.argmax(probabilities)

    sentiment_label = "Positive" if prediction == 1 else "Negative"
    confidence = float(probabilities[prediction])

    return {
        "text": text,
        "prediction": int(prediction),
        "sentiment": sentiment_label,
        "confidence": confidence,
        "prob_negative": float(probabilities[0]),
        "prob_positive": float(probabilities[1])
    }

# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT Sentiment Analysis Assignment")
    parser.add_argument("--train", action="store_true", default=True, help="Run training pipeline")
    parser.add_argument("--predict", type=str, default=None, help="Custom sentence to predict sentiment")
    args = parser.parse_args()

    if args.predict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = CONFIG['output_dir'] if os.path.exists(CONFIG['output_dir']) else CONFIG['model_name']
        print(f"Loading model from {model_path} for inference...")
        tok = BertTokenizer.from_pretrained(model_path)
        mdl = BertForSequenceClassification.from_pretrained(model_path).to(device)
        result = predict_sentiment(args.predict, mdl, tok, device)
        print(f"\nInput: \"{result['text']}\"")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']*100:.2f}%)")
    else:
        run_training_pipeline()
