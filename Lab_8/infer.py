"""
=============================================================================
Deep Learning Lab - Assignment 8
Interactive Real-Time Inference Script with Fine-Tuned BERT
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

import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_DIR = "saved_model"
MAX_LEN = 128

def load_sentiment_model(model_dir=MODEL_DIR):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found. Please train the model first.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading fine-tuned BERT model from '{model_dir}' onto {device}...")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model, device

def predict_sentiment(text, tokenizer, model, device):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
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
        prediction = int(np.argmax(probabilities))

    label_map = {0: "Negative", 1: "Positive"}
    return {
        "text": text,
        "sentiment": label_map[prediction],
        "confidence": float(probabilities[prediction]),
        "neg_prob": float(probabilities[0]),
        "pos_prob": float(probabilities[1])
    }

def main():
    tokenizer, model, device = load_sentiment_model()
    print("\n" + "=" * 65)
    print("  Pretrained BERT Sentiment Analysis - Interactive Testing")
    print("  Type any review or sentence to classify. (Type 'exit' to quit)")
    print("=" * 65 + "\n")

    test_samples = [
        "The movie was an absolute delight with brilliant performances!",
        "Terrible experience, completely broke on the first day.",
        "Not bad at all, actually exceeded my expectations.",
        "I was so excited to buy this, but the quality is shockingly awful."
    ]

    print("--- Running Default Validation Examples ---")
    for sample in test_samples:
        res = predict_sentiment(sample, tokenizer, model, device)
        symbol = "[+]" if res['sentiment'] == "Positive" else "[-]"
        print(f"{symbol} Review: \"{res['text']}\"")
        print(f"    Sentiment: {res['sentiment']} | Confidence: {res['confidence']*100:.2f}% (Pos: {res['pos_prob']*100:.1f}%, Neg: {res['neg_prob']*100:.1f}%)\n")

    if len(sys.argv) > 1:
        custom_input = " ".join(sys.argv[1:])
        res = predict_sentiment(custom_input, tokenizer, model, device)
        print(f"\nCustom Input: \"{res['text']}\"")
        print(f"Sentiment: {res['sentiment']} (Confidence: {res['confidence']*100:.2f}%)")

if __name__ == "__main__":
    main()
