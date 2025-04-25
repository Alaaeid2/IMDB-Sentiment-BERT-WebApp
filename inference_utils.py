import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import re
import numpy as np
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from nltk.corpus import words
import nltk
nltk.download('words')

MODEL_PATH = "models/fine_tuned_bert"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)

english_vocab = set(words.words())

def is_valid_input(text, min_word_count=2, min_real_english=2, min_unique=2):
    cleaned = re.sub(r"[^\w\s]", "", text)
    cleaned = re.sub(r"\d+", "", cleaned)
    words_list = cleaned.lower().strip().split()
    meaningful_words = [w for w in words_list if len(w) >= 3 and w.isalpha()]
    real_words = [w for w in meaningful_words if w in english_vocab]
    unique_words = set(meaningful_words)

    return (
        len(meaningful_words) >= min_word_count
        and len(real_words) >= min_real_english
        and len(unique_words) >= min_unique
    )

def predict_sentiment(text):
    if not is_valid_input(text):
        return {
            "text": text,
            "sentiment": "Invalid Input",
            "confidence": 0.0
        }

    encodings = tokenizer(
        [text],
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='tf'
    )
    outputs = model(encodings)
    probs = tf.nn.softmax(outputs.logits, axis=1).numpy()
    pred = np.argmax(probs)
    confidence = float(np.max(probs))
    label_map = {0: "Negative", 1: "Positive"}

    return {
        "text": text,
        "sentiment": label_map[pred],
        "confidence": round(confidence, 4)
    }
