import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

# Static paths
DATA_PATH = "data/processed_data.npz"
CSV_DATA_PATH = "data/IMDB_Dataset.csv"  # Same CSV used for fine-tuning BERT

def load_data(model_name):
    if model_name == "bert":
        # Load and prepare data for BERT using the same CSV and tokenizer
        df = pd.read_csv(CSV_DATA_PATH)
        label_map = {'negative': 0, 'positive': 1}
        df['sentiment'] = df['sentiment'].map(label_map)
        if df['sentiment'].isnull().any():
            raise ValueError("Found sentiment labels outside of 'positive'/'negative' in CSV.")

        # For evaluation, take the last 10,000 samples (or adjust as needed)
        test_df = df[-10000:].copy()
        X_test, y_test = prepare_bert_data(test_df)
    else:
        # Load data for LSTM and hybrid models
        data = np.load(DATA_PATH)
        X_test, y_test = data['X_test'], data['y_test']
    
    return X_test, y_test

def prepare_bert_data(df, max_len=128):
    # Load the tokenizer from the local fine-tuned directory
    tokenizer = BertTokenizer.from_pretrained('models/fine_tuned_bert')
    encodings = tokenizer(
        df.review.tolist(),
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='tf'
    )
    y_test = np.array(df.sentiment.values, dtype=np.int32)
    return dict(encodings), y_test

def evaluate(model_name):
    model_map = {
        "lstm": {
            "model_path": "models/lstm_model_tuned.h5",
            "report_path": "evaluation/lstm_report.txt",
            "conf_matrix_path": "evaluation/lstm_confusion_matrix1.png"
        },
        "hybrid": {
            "model_path": "models/Conv1D-BiLSTM_model.h5",
            "report_path": "evaluation/Conv1D-BiLSTM_report.txt",
            "conf_matrix_path": "evaluation/Conv1D-BiLSTM_confusion_matrix.png"
        },
        "bert": {
            "model_path": "models/fine_tuned_bert",  # Local path
            "report_path": "evaluation/bert_report.txt",
            "conf_matrix_path": "evaluation/bert_confusion_matrix.png"
        }
    }

    if model_name not in model_map:
        print(f"Model '{model_name}' not found. Choose from: {list(model_map.keys())}")
        return

    paths = model_map[model_name]
    model_path = paths["model_path"]
    report_path = paths["report_path"]
    conf_matrix_path = paths["conf_matrix_path"]

    print(f"Evaluating: {model_name.upper()}")

    # Load model (BERT might need custom load logic)
    if model_name == "bert":
        model = TFBertForSequenceClassification.from_pretrained(model_path)
    else:
        model = load_model(model_path)

    # Load data specific to the model
    X_test, y_test = load_data(model_name)

    # Predict
    if model_name == "bert":
        predictions = model.predict(X_test, batch_size=16)  # Match fine-tuning batch size
        logits = predictions.logits
        probabilities = tf.sigmoid(logits)
        y_pred = np.argmax(probabilities, axis=1)
    else:
        y_pred = (model.predict(X_test) > 0.5).astype("int32")

    # Ensure y_test is in the correct shape
    y_test = np.squeeze(y_test)

    # Classification Report
    report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"])
    print("\nClassification Report:")
    print(report)

    # Save Report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"{model_name.upper()} Classification Report\n")
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"Confusion Matrix - {model_name.upper()} Model")
    plt.savefig(conf_matrix_path)
    plt.close()

    print(f"Report saved to {report_path}")
    print(f"Confusion matrix saved to {conf_matrix_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sentiment model (LSTM, Conv1D-BiLSTM, or BERT)")
    parser.add_argument("model", choices=["lstm", "hybrid", "bert"], help="Model name to evaluate")
    args = parser.parse_args()

    evaluate(args.model)