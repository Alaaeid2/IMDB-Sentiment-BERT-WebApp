import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np
import re
import string
import argparse
import nltk
from nltk.corpus import words
from transformers import logging
logging.set_verbosity_error()

# Load tokenizer and model (ensure they're fine-tuned and saved locally)
MODEL_PATH = "models/fine_tuned_bert"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)

#nltk.download('words')
english_vocab = set(words.words())

def is_valid_input(text, min_word_count=3, min_real_english=2, min_unique=3):
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
def predict_sentiment(texts, max_len=128):
    if isinstance(texts, str):
        texts = [texts]

    results = []
    for text in texts:
        if not is_valid_input(text):
            print(f"\nText: {text}")
            print("Input seems invalid or too vague. Please enter meaningful English text.")
            continue

        # now we know it’s valid—run the model
        encodings = tokenizer(
            [text], padding='max_length', truncation=True, max_length=max_len, return_tensors='tf'
        )
        outputs = model(encodings)
        logits = outputs.logits
        probs = tf.nn.softmax(logits, axis=1).numpy()
        pred = np.argmax(probs, axis=1)[0]
        prob = float(np.max(probs))
        label_map = {0: "Negative", 1: "Positive"}
        results.append((text, label_map[pred], prob))

    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Text to analyze")
    args = parser.parse_args()

    results = predict_sentiment(args.text)
    for text, label, prob in results:
        print(f"\nText: {text}\n🔮 Sentiment: {label} ({prob*100:.2f}%)")
