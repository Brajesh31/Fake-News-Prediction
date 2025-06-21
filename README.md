# Fake News Detection Model using TensorFlow

Detect fake news using a deep learning model built with TensorFlow in Python. This model classifies news articles as **FAKE** or **REAL** based on the text content, helping to prevent misinformation spread and protect public trust.

---

##  Project Overview

Fake news can mislead readers and influence public opinion negatively. This project uses NLP techniques and a deep learning LSTM model with pre-trained GloVe embeddings to detect fake news from textual data.

Key features:
- Text preprocessing and tokenization
- Use of pre-trained GloVe embeddings for better word representation
- LSTM architecture for sequence modeling
- Binary classification (FAKE vs REAL news)

---

##  Dataset

The dataset consists of news articles with their corresponding labels (FAKE or REAL).

- Contains columns: `title`, `text`, `label`
- Source: [Fake News Dataset](https://www.kaggle.com/datasets)

---

##  Tech Stack

- Python  
- TensorFlow & Keras  
- NumPy, Pandas  
- Scikit-learn  
- Pre-trained GloVe embeddings  

---

##  Model Architecture

- Embedding layer initialized with GloVe embeddings (50-dimensional vectors)
- Conv1D layer for pattern detection
- MaxPooling1D layer
- LSTM layer (64 units) for sequence learning
- Dense output layer with sigmoid activation for binary classification

---
