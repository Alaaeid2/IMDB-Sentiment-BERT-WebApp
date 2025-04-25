import numpy as np
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Enable memory growth for GPU (avoid OOM errors)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# Paths
DATA_PATH = "data/processed_data.npz"
TOKENIZER_PATH = "preprocessing/tokenizer.pkl"
MODEL_PATH = "models/lstm_model_tuned.h5"

# Hyperparameters (Tuned)
VOCAB_SIZE = 5000
EMBEDDING_DIM = 128       # More embedding info
LSTM_UNITS = 128
MAX_LEN = 200
EPOCHS = 8                # More time to learn
BATCH_SIZE = 64
LEARNING_RATE = 2e-5      # Smaller LR for better tuning

def load_data():
    data = np.load(DATA_PATH)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']

def load_tokenizer():
    with open(TOKENIZER_PATH, 'rb') as f:
        return pickle.load(f)

def build_model():
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(LSTM_UNITS, return_sequences=True),    # Stacking LSTMs
        Dropout(0.3),
        LSTM(LSTM_UNITS // 2),                      # Second LSTM layer
        Dropout(0.3),
        Dense(64, activation='relu'),               # Dense layer before final
        Dropout(0.5),
        Dense(1, activation='sigmoid')              # Binary classification
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
    return model

def train():
    X_train, X_test, y_train, y_test = load_data()
    model = build_model()

    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max')

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint],
        verbose=2
    )
    print(f"Tuned model trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
