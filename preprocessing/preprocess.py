import pandas as pd
import re
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Parameters
VOCAB_SIZE = 5000
MAX_LEN = 200
TOKENIZER_PATH = "preprocessing/tokenizer.pkl"
OUTPUT_PATH = "data/processed_data.npz"

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric
    return text.lower().strip()

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df['review'] = df['review'].apply(clean_text)
    df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    X = df['review'].values
    y = df['label'].values

    # Tokenization
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)

    sequences = tokenizer.texts_to_sequences(X)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    # Save tokenizer
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(padded, y, test_size=0.2, random_state=42)

    # Save preprocessed data
    np.savez_compressed(OUTPUT_PATH, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    print(f"Preprocessing done. Data saved to {OUTPUT_PATH}, tokenizer to {TOKENIZER_PATH}")

if __name__ == "__main__":
    csv_path = "data/IMDB_Dataset.csv"
    load_and_preprocess(csv_path)
