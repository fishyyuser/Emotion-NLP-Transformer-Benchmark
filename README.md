EMOTION NLP TRANSFORMER BENCHMARK — PROJECT README
(BiLSTM + Attention vs BERT)
=======================================================

This repository contains a full end-to-end benchmark and
comparison of two fundamentally different NLP approaches
for multi-class emotion classification:

1. Classical Deep Learning
   → BiLSTM + Custom Attention (TensorFlow)

2. Modern Transformer-Based Learning
   → BERT / DistilBERT (Hugging Face + PyTorch)

The goal of this project is NOT just to maximize accuracy,
but to study:

- Accuracy vs Inference Speed
- Classical DL vs Transformers
- Error Patterns across Emotion Classes
- Real-World Deployment Trade-offs

## PROBLEM STATEMENT

Task:
Multi-class emotion classification from short text.

Emotion Classes:

- 0 → sadness
- 1 → joy
- 2 → love
- 3 → anger
- 4 → fear
- 5 → surprise

Each sample represents a real-world short text
(similar to social media, chat messages, etc).

## MODELS USED

1. BiLSTM + Attention (TensorFlow)

   - Custom Attention Layer
   - Word-level Tokenization
   - Lightweight & Fast Inference
   - Designed for Real-Time Deployment

2. BERT / DistilBERT (Transformers)
   - Pretrained Transformer Encoder
   - Subword Tokenization (WordPiece)
   - High Semantic Understanding
   - Computationally Heavy

## MODEL SOURCE — BiLSTM + ATTENTION (ORIGINAL REPOSITORY)

The BiLSTM + Attention model used in this benchmark was
originally trained and developed in my previous repository:

https://github.com/fishyyuser/Text-Emotion-Predictor

In THIS repository:

- The pretrained **bilstm_attention_emotions.keras** model is directly reused
- Only baseline evaluation is performed
- The model is then benchmarked against BERT
- The same trained weights are used for the Streamlit frontend

This ensures that:

- BiLSTM performance represents a true frozen baseline
- Comparisons against BERT are fair and unbiased
- No additional tuning is applied to the BiLSTM model

## EVALUATION METRICS USED

- Accuracy
- Macro F1 (Class Balance Sensitivity)
- Weighted F1 (Overall Performance)
- Confusion Matrix
- Inference Time on Full Test Set

## FINAL BENCHMARK RESULTS (SUMMARY)

BiLSTM + Attention:

- Accuracy ≈ 95.5%
- Macro F1 ≈ 93.3%
- Weighted F1 ≈ 95.6%
- Inference Time ≈ 2.6 seconds on full test set

BERT / DistilBERT:

- Accuracy ≈ 96.8%
- Macro F1 ≈ 94.9%
- Weighted F1 ≈ 96.8%
- Inference Time ≈ 82 seconds on full test set

## KEY ENGINEERING CONCLUSION

- BERT provides slightly better semantic understanding
  and class balance handling.

- BiLSTM + Attention is ~30x faster during inference.

- For real-time systems:
  → BiLSTM is the better choice.

- For offline analytics and deeper language reasoning:
  → BERT is the better choice.

This project demonstrates that:
The best model is decided by deployment constraints,
not just accuracy numbers.

## PROJECT STRUCTURE

```bash
EMOTION-NLP-TRANSFORMER-BENCHMARK/
|
|-- artifacts/
|   |
|   |-- model/
|       |
|       |-- bilstm_attention_emotional_model/
|       |   |-- bilstm_attention_emotions.keras
|       |   |-- tokenizer.pkl
|       |
|       |-- bert_emotion_model/
|           |-- config.json
|           |-- model.safetensors / pytorch_model.bin
|           |-- tokenizer.json
|           |-- tokenizer_config.json
|           |-- vocab.txt
|
|-- notebooks/
|   |
|   |-- 01_bilstm_baseline_evaluation.ipynb
|   |-- 02_bert_training_and_evaluation.ipynb
|   |-- 03_bilstm_vs_bert_comparison.ipynb
|
|-- streamlit_app.py
|-- requirements.txt
|-- LICENSE
|-- README.md
```

## STREAMLIT FRONTEND

The Streamlit app provides:

- Side-by-side prediction
- Emoji-based emotion display
- Per-class probability distribution
- Real-time inference latency comparison

Run locally using:

```bash
streamlit run streamlit_app.py
```

## WHAT THIS DEMONSTRATES

This repository demonstrates:

- Classical NLP Deep Learning
- Transformer Fine-Tuning
- Model Benchmarking Methodology
- Deployment-Oriented Thinking
- Latency vs Accuracy Trade-off Analysis
- Error Pattern Analysis using Confusion Matrices

## FINAL NOTE

This project represents a complete NLP engineering cycle:

Dataset → Training → Evaluation → Benchmarking →
Deployment → Real-Time Comparison
