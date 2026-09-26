# Deep Learning Lab - Assignment 8: Pretrained BERT for Sentiment Analysis

This repository contains the complete implementation, training pipeline, evaluation metrics, and documentation for **Assignment 8: Implement Pretrained BERT Model for Sentiment Analysis / Text Classification**.

---

## 📁 Repository Structure

```
Lab_8/
├── data/
│   └── sentiment_dataset.csv                  # Balanced dataset (198 samples: Positive & Negative)
├── saved_model/                               # Best fine-tuned BERT checkpoint
│   ├── config.json
│   ├── model.safetensors                      # Fine-tuned model weights (383 MB)
│   ├── tokenizer_config.json
│   └── vocab.txt
├── evaluation_plots/                          # High-resolution performance charts
│   ├── confusion_matrix.png                   # Test set confusion matrix
│   └── training_curves.png                    # Loss and accuracy vs. epochs
├── bert_sentiment_classifier.py               # Complete training & evaluation pipeline script
├── infer.py                                   # Real-time CLI inference tool
├── create_dataset.py                          # Dataset generation script
├── build_notebook.py                          # Automated notebook builder
├── Assignment_8_BERT_Sentiment_Analysis.ipynb # Jupyter Notebook with code, theory & plots
├── LAB_REPORT_ASSIGNMENT_8.md                 # Complete formal lab report & Viva Voce Q&A
└── README.md                                  # Project overview and quick start guide
```

---

## 🚀 Quick Start Guide

### 1. Launch Interactive Modern Web Studio (Frontend + Backend)
```bash
python server.py
```
Open **[http://localhost:8080](http://localhost:8080)** in any browser.
Features:
- ✨ Real-time bidirectional BERT sentiment inference.
- 🔬 Interactive WordPiece Subword Tokenizer Inspector showing `[CLS]`, `[SEP]`, `##tokens`, and vocabulary IDs.
- 📊 Live Softmax probability gauges for Positive and Negative sentiment.
- ⚡ Sub-millisecond GPU inference tracking on NVIDIA GeForce RTX 2050.
- 📈 Dedicated Model Diagnostics tab with inline Confusion Matrix and Loss/Accuracy plots.

### 2. Run Complete Training and Evaluation Pipeline
```bash
python bert_sentiment_classifier.py
```
This trains `bert-base-uncased` for 3 epochs using CUDA GPU acceleration, evaluates on the hold-out test set, saves the best checkpoint to `saved_model/`, and produces diagnostic plots in `evaluation_plots/`.

### 3. Run Interactive Real-Time Prediction via CLI
```bash
python infer.py "The movie was an absolute masterpiece with stellar performances!"
```

### 4. Open Jupyter Notebook
```bash
jupyter notebook Assignment_8_BERT_Sentiment_Analysis.ipynb
```

---

## 📊 Summary of Results

- **Model:** `bert-base-uncased` (110 Million Parameters)
- **Accelerator:** NVIDIA GeForce RTX 2050 (CUDA 12.1)
- **Training Epochs:** 3
- **Test Accuracy:** **90.00%**
- **Test Precision:** **90.18%**
- **Test Recall:** **90.00%**
- **Test F1-Score:** **89.99%**
