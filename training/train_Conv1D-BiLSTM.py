import numpy as np
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

# GPU memory growth (optional)
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
MODEL_PATH = "models/Conv1D-BiLSTM_model.h5"

# Hyperparams
VOCAB_SIZE = 5000
EMBEDDING_DIM = 128
MAX_LEN = 200
EPOCHS = 6
BATCH_SIZE = 64
LEARNING_RATE = 2e-4

def load_data():
    data = np.load(DATA_PATH)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']

def build_model():
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(64)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
    return model

def train():
    X_train, X_test, y_train, y_test = load_data()
    model = build_model()

    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max')

    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=[checkpoint],
              verbose=2)
    
    print(f"Conv1D-BiLSTM model trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
