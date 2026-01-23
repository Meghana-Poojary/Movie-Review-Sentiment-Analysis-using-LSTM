# 🎬 Movie Review Sentiment Analysis (IMDB)

This project implements a **binary sentiment analysis model** that classifies movie reviews as **positive or negative** using a **Long Short-Term Memory (LSTM)** neural network trained on the **IMDB movie reviews dataset**.

The model is built with **TensorFlow / Keras** and focuses on learning sequential patterns in text data to capture contextual sentiment.

---

## 📌 Project Overview

* **Task**: Binary sentiment classification
* **Dataset**: IMDB Movie Reviews (50,000 reviews)
* **Classes**: Positive / Negative
* **Model**: Embedding → LSTM → Dense
* **Framework**: TensorFlow (Keras API)

---

## 🧠 Model Architecture

```
Embedding Layer (10,000 words, 128 dimensions)
        ↓
LSTM Layer (64 units)
        ↓
Dense Layer (1 unit, Sigmoid activation)
```

### Why LSTM?

* Captures **long-term dependencies** in text
* Handles variable-length sequences
* Suitable for sentiment where word order matters

---

## 📂 Dataset Details

* **Source**: Keras IMDB dataset
* **Vocabulary size**: 10,000 most frequent words
* **Training samples**: 25,000
* **Testing samples**: 25,000
* **Labels**:

  * `0` → Negative review
  * `1` → Positive review

---

## ⚙️ Preprocessing Steps

1. Load IMDB dataset with a fixed vocabulary size
2. Convert word indices into padded sequences
3. Pad / truncate all reviews to a fixed length of **500 tokens**

```python
pad_sequences(..., maxlen=500, padding='post', truncating='post')
```

---

## 🚀 Training Configuration

| Parameter         | Value               |
| ----------------- | ------------------- |
| Optimizer         | Adam                |
| Learning Rate     | 0.001               |
| Gradient Clipping | clipnorm = 0.1      |
| Loss Function     | Binary Crossentropy |
| Batch Size        | 32                  |
| Epochs            | 20                  |
| Validation Split  | 20%                 |

Gradient clipping is used to **prevent exploding gradients**, improving training stability.

Typical results:

* **Test Accuracy**: 0.849839985370636
* **Test Loss**: 0.7147225141525269 (varies by run)
---

## 📌 Key Learnings

* How word embeddings represent text numerically
* How LSTM networks process sequential data
* Importance of gradient clipping in RNN-based models
* End-to-end NLP pipeline using TensorFlow

---

## 🔮 Future Improvements

* Use **Bidirectional LSTM**
* Add **Dropout** for regularization
* Replace IMDB tokenizer with a custom `Tokenizer`
* Deploy model using **Streamlit**
* Add inference on real-world text reviews

---

## 👩‍💻 Author

**Meghana Poojary**
Final Year Student | Machine Learning & NLP Enthusiast

---

## This app is live on
https://movie-review-sentiment-analysis-using-lstm-txgk39mtwydiyfofczv.streamlit.app/

---

## ⭐ Acknowledgements

* TensorFlow / Keras
* IMDB Movie Review Dataset

---
