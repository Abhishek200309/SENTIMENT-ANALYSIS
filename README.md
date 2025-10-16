# SENTIMENT-ANALYSIS

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: ABHISHEK VICTOR RAJ MANESH

*INTERN ID*: CT12DY2725

*DOMAIN*: DATA ANALYTICS

*DURATION*: 12 WEEKS

*MENTOR*: NEELA SANTOSH

# Description

This project focuses on performing sentiment analysis on textual data such as tweets and reviews. The goal is to classify each text into two categories — Positive or Negative — using Natural Language Processing (NLP) and Deep Learning techniques.

The project demonstrates a complete end-to-end workflow in NLP:

  - Text data preprocessing and cleaning

  - Tokenization and sequence padding

  - Model building using Bidirectional LSTM

  - Model training and evaluation

  - WordCloud visualization of sentiments

  - Real-time prediction through a Streamlit web app

By following this process, we gain practical experience in text preprocessing, sequence modeling, and sentiment classification — key skills in real-world NLP applications.

# Workflow Explanation

1. Data Loading

  The dataset (Twitter Sentiment Dataset) contains labeled tweets categorized into four classes: Positive, Negative, Neutral, and Irrelevant.
  It is loaded using Pandas for preprocessing and analysis.


2. Data Preprocessing

  Since text data is often noisy, several preprocessing steps are applied:

  - Lowercasing all text

  - Removing special characters, URLs, mentions, and punctuation

  - Tokenizing the text into individual words

  - Removing stopwords using NLTK

  - Padding sequences to ensure uniform input length for the model

These steps transform raw tweets into clean, numeric sequences suitable for deep learning models.


3. Tokenization and Sequence Preparation

- The Keras Tokenizer converts words into integer indices.

- Sequences are padded to a fixed length using pad_sequences.

- The dataset is split into training and validation sets (typically 80–20).


4. Model Building (Bidirectional LSTM)

A Bidirectional Long Short-Term Memory (BiLSTM) model is built to capture both past and future context in the text.
The architecture includes:

  - Embedding layer – Converts word indices to dense vector representations

  - Bidirectional LSTM layer – Learns contextual dependencies

  - Dropout layer – Prevents overfitting

  - Dense output layer – Softmax activation for multi-class classification

This architecture enables the model to understand sentiment nuances effectively.


5. Model Training and Evaluation

- Loss Function: Categorical Crossentropy

- Optimizer: Adam

- Metrics: Accuracy

An EarlyStopping callback is used to prevent overfitting by stopping training when validation loss no longer improves.

The model achieves a strong validation accuracy, indicating effective sentiment classification.


6. Visualization

To better understand the data and model results, the following visualizations are generated:

  - WordClouds – Highlight the most frequent words in positive and negative tweets

  - Training Accuracy Plot – Shows improvement of model performance over epochs

  - Confusion Matrix (optional) – To evaluate predictions across sentiment classes


7. Streamlit Web Application

A simple Streamlit app is created for real-time sentiment prediction.
Features include:

  - Text input box for entering custom statements

  - Predict button for classification

  - Displays sentiment label (Positive, Negative, Neutral, or Irrelevant) along with confidence score

  - Loads pre-trained model (sentiment_model.h5) and tokenizer for instant predictions

# Results

<img width="1919" height="875" alt="Image" src="https://github.com/user-attachments/assets/b208556d-86ba-42b6-bf26-f67b7b96072c" />

<img width="1919" height="876" alt="Image" src="https://github.com/user-attachments/assets/3d27be33-1883-43df-9feb-40de055d9732" />

# Conclusion

This project demonstrates how Natural Language Processing (NLP) and Deep Learning can be applied to extract sentiments from textual data. From cleaning tweets to deploying a prediction app, it provides a complete practical pipeline for sentiment analysis.

By combining data preprocessing, model training, visualization, and deployment, this project serves as a strong foundation for building intelligent systems that understand human emotions through text.
