"""
=============================================================================
Deep Learning Lab - Assignment 8
FastAPI Backend Server for BERT Sentiment Analysis Web UI
=============================================================================
"""

import os
import sys

# Prevent transformers from loading conflicting tensorflow protobuf bindings
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.modules['tensorflow'] = None

import gc
import time
import argparse
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import uvicorn

MODEL_DIR = "saved_model"
MAX_LEN = 128

# 1. Initialize FastAPI Application
app = FastAPI(
    title="BERT Sentiment Analysis AI Studio",
    description="Interactive Web UI API for Deep Learning Lab Assignment 8",
    version="1.0.0"
)

# 2. Mount Static Files
os.makedirs("static", exist_ok=True)
os.makedirs("evaluation_plots", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/evaluation_plots", StaticFiles(directory="evaluation_plots"), name="evaluation_plots")

# 3. Model & Device Global State
tokenizer = None
model = None
device = None

def init_model():
    """Safely loads model onto GPU (with CPU fallback) and manages memory."""
    global tokenizer, model, device
    if model is not None and tokenizer is not None:
        return tokenizer, model, device

    # Determine device
    if torch.cuda.is_available():
        try:
            # Test simple tensor allocation to verify GPU memory availability
            test_t = torch.zeros((1, 1), device="cuda")
            del test_t
            torch.cuda.empty_cache()
            device = torch.device("cuda")
        except Exception:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    model_path = MODEL_DIR if os.path.exists(MODEL_DIR) else "bert-base-uncased"
    print(f"\n[BERT Backend] Loading model from '{model_path}' onto {device}...")

    # Garbage collection before loading to free commit limit
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    try:
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
        model.to(device)
    except Exception as e:
        print(f"[BERT Backend] GPU load failed ({e}), falling back to CPU...")
        device = torch.device("cpu")
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
        model.to(device)

    model.eval()
    print(f"[BERT Backend] Model loaded successfully on {device}!\n")
    return tokenizer, model, device

# 4. Request / Response Schemas
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    prob_negative: float
    prob_positive: float
    tokens: list[str]
    token_ids: list[int]
    latency_ms: float

# 5. Routes
@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

@app.get("/api/info")
async def get_system_info():
    tok, mdl, dev = init_model()
    dev_name = torch.cuda.get_device_name(0) if dev.type == "cuda" and torch.cuda.is_available() else "CPU"
    return {
        "status": "online",
        "device": str(dev),
        "device_name": dev_name,
        "model_name": "bert-base-uncased",
        "parameters": sum(p.numel() for p in mdl.parameters()),
        "max_length": MAX_LEN,
        "classes": ["Negative (0)", "Positive (1)"]
    }

@app.post("/api/predict", response_model=SentimentResponse)
async def predict_sentiment(req: SentimentRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    tok, mdl, dev = init_model()

    start_time = time.perf_counter()

    encoding = tok.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(dev)
    attention_mask = encoding["attention_mask"].to(dev)
    token_type_ids = encoding["token_type_ids"].to(dev)

    with torch.no_grad():
        outputs = mdl(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        prediction = int(np.argmax(probabilities))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    raw_token_ids = encoding["input_ids"][0].tolist()
    non_pad_ids = [tid for tid in raw_token_ids if tid != tok.pad_token_id]
    subword_tokens = tok.convert_ids_to_tokens(non_pad_ids)

    sentiment_label = "Positive" if prediction == 1 else "Negative"
    confidence = float(probabilities[prediction])

    return SentimentResponse(
        text=text,
        sentiment=sentiment_label,
        confidence=round(confidence, 4),
        prob_negative=round(float(probabilities[0]), 4),
        prob_positive=round(float(probabilities[1]), 4),
        tokens=subword_tokens,
        token_ids=non_pad_ids,
        latency_ms=round(elapsed_ms, 2)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start BERT Web Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to run web server on")
    args = parser.parse_args()

    port = args.port
    print(f"===============================================================")
    print(f"  BERT Sentiment Analysis Web Studio")
    print(f"  Access URL: http://localhost:{port}")
    print(f"===============================================================")
    
    # Initialize model once before server starts
    init_model()

    # Pass the app object directly (not string "server:app") to avoid double import
    uvicorn.run(app, host="127.0.0.1", port=port, reload=False)
