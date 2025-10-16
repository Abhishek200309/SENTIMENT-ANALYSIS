import os
import re
import pickle
import numpy as np
import pandas as pd
import nltk
import streamlit as st
import tensorflow as tf
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# 1) Setup
# -------------------------
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
MAX_LEN = 50
VOCAB_SIZE = 10000

MODEL_PATH = "sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"

# -------------------------
# 2) Load Dataset (only if training needed)
# -------------------------
def load_and_preprocess_data():
    file_path = "D:/197001__/twitter_training.csv/twitter_training.csv"   # Update path if needed
    df = pd.read_csv(file_path, header=None, names=["id", "entity", "label", "text"])
    df = df[["text", "label"]].dropna()

    # Map labels
    label_map = {"Negative": 0, "Positive": 1, "Neutral": 2, "Irrelevant": 3}
    df["label"] = df["label"].map(label_map)

    # Clean text
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        text = " ".join([w for w in text.split() if w not in STOPWORDS])
        return text

    df["clean_text"] = df["text"].astype(str).apply(clean_text)

    return df

# -------------------------
# 3) Training Function
# -------------------------
def train_and_save_model():
    df = load_and_preprocess_data()

    # Tokenization
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["clean_text"])
    sequences = tokenizer.texts_to_sequences(df["clean_text"])
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
    y = df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = Sequential([
        Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(4, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=5,
                        batch_size=64,
                        callbacks=[es],
                        verbose=1)

    # Save model & tokenizer
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    return model, tokenizer, history

# -------------------------
# 4) Load or Train
# -------------------------
if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    history = None
else:
    model, tokenizer, history = train_and_save_model()

# -------------------------
# 5) Streamlit Web App
# -------------------------
st.title("🚀 Twitter Sentiment Analysis")
st.write("Predict if a text is **Positive, Negative, Neutral, or Irrelevant**")

label_map_rev = {0: "Negative", 1: "Positive", 2: "Neutral", 3: "Irrelevant"}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join([w for w in text.split() if w not in STOPWORDS])
    return text

def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([clean_text(text)])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    pred = model.predict(pad)
    label = label_map_rev[np.argmax(pred)]
    return label, float(np.max(pred))

# User input
user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input.strip():
        label, confidence = predict_sentiment(user_input)
        st.success(f"Prediction: **{label}** (Confidence: {confidence:.2f})")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Show training chart only if training just happened
if history:
    st.line_chart({
        "Training Accuracy": history.history["accuracy"],
        "Validation Accuracy": history.history["val_accuracy"]
    })
