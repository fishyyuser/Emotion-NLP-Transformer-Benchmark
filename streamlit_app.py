import os
import re
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# Paths & Config
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

BILSTM_MODEL_PATH = MODEL_DIR / "bilstm_attention_emotional_model/bilstm_attention_emotions.keras"
TOKENIZER_PATH = MODEL_DIR / "bilstm_attention_emotional_model/tokenizer.pkl"

BERT_MODEL_DIR = MODEL_DIR / "bert_emotion_model"

MAX_LEN = 80

ID2LABEL = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

LABEL2EMOJI = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲",
}

# ============================================================
# Text Cleaning
# ============================================================

url_pattern = re.compile(r"http\S+|www\.\S+")
mention_pattern = re.compile(r"@\w+")
space_pattern = re.compile(r"\s+")

def clean_text(s: str) -> str:
    s = str(s).lower()
    s = url_pattern.sub(" ", s)
    s = mention_pattern.sub(" ", s)
    s = space_pattern.sub(" ", s)
    return s.strip()

# ============================================================
# Custom Attention Layer (For BiLSTM)
# ============================================================

class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        score = tf.matmul(inputs, self.W)
        score = tf.squeeze(score, axis=-1)
        alpha = tf.nn.softmax(score, axis=1)
        alpha = tf.expand_dims(alpha, axis=-1)
        context = tf.reduce_sum(inputs * alpha, axis=1)
        return context

# ============================================================
# Loaders
# ============================================================

@st.cache_resource
def load_bilstm():
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    model = tf.keras.models.load_model(
        BILSTM_MODEL_PATH,
        custom_objects={"Attention": Attention}
    )
    return model, tokenizer


@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    model.eval()
    return model, tokenizer


# ============================================================
# Prediction Functions
# ============================================================

def predict_bilstm(text, model, tokenizer):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    start = time.time()
    probs = model.predict(pad, verbose=0)[0]
    latency = time.time() - start

    pred_id = int(np.argmax(probs))
    label = ID2LABEL[pred_id]

    return label, probs, latency


def predict_bert(text, model, tokenizer):
    cleaned = clean_text(text)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    latency = time.time() - start

    pred_id = int(np.argmax(probs))
    label = ID2LABEL[pred_id]

    return label, probs, latency

# ============================================================
# UI
# ============================================================

st.set_page_config(page_title="BiLSTM vs BERT Emotion Classifier", page_icon="🧠")
st.title("🧠 Emotion Classification — BiLSTM vs BERT")

st.markdown(
"""
Compare **Classical Deep Learning (BiLSTM + Attention)** vs  
**Transformer (BERT)** side-by-side in real time.
"""
)

with st.spinner("Loading models..."):
    bilstm_model, bilstm_tokenizer = load_bilstm()
    bert_model, bert_tokenizer = load_bert()

user_text = st.text_area(
    "Enter your text:",
    value="I can't believe how amazing today has been!",
    height=120
)

if st.button("🔍 Analyze Emotion", type="primary"):
    if not user_text.strip():
        st.warning("Enter some text first.")
    else:
        bilstm_label, bilstm_probs, bilstm_time = predict_bilstm(
            user_text, bilstm_model, bilstm_tokenizer
        )

        bert_label, bert_probs, bert_time = predict_bert(
            user_text, bert_model, bert_tokenizer
        )

        col1, col2 = st.columns(2)

        # ================= BiLSTM =================
        with col1:
            st.subheader("BiLSTM + Attention")
            st.markdown(
                f"""
### {LABEL2EMOJI[bilstm_label]} **{bilstm_label.upper()}**  
**Inference Time:** `{bilstm_time:.4f} sec`
"""
            )

            df_bilstm = pd.DataFrame({
                "Emotion": [f"{LABEL2EMOJI[ID2LABEL[i]]} {ID2LABEL[i]}" for i in range(6)],
                "Probability": bilstm_probs.tolist()
            })

            st.bar_chart(df_bilstm, x="Emotion", y="Probability")

        # ================= BERT =================
        with col2:
            st.subheader("BERT (Transformer)")
            st.markdown(
                f"""
### {LABEL2EMOJI[bert_label]} **{bert_label.upper()}**  
**Inference Time:** `{bert_time:.4f} sec`
"""
            )

            df_bert = pd.DataFrame({
                "Emotion": [f"{LABEL2EMOJI[ID2LABEL[i]]} {ID2LABEL[i]}" for i in range(6)],
                "Probability": bert_probs.tolist()
            })

            st.bar_chart(df_bert, x="Emotion", y="Probability")

else:
    st.info("Enter text and click **Analyze Emotion** to compare both models.")
