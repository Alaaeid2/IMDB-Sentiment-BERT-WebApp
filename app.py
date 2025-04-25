import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np
from inference_utils import predict_sentiment

app = Flask(__name__)

# Load model and tokenizer once
MODEL_PATH = "models/fine_tuned_bert"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)

# Web UI Route
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        user_input = request.form["text"]
        prediction = predict_sentiment(user_input)
        text, sentiment, confidence = prediction["text"], prediction["sentiment"], prediction["confidence"]
        result = {
            "text": text,
            "sentiment": sentiment,
            "confidence": f"{confidence*100:.2f}%"
        }
    return render_template("index.html", result=result)

# API Route
@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    text = data.get("text", "")
    prediction = predict_sentiment(text)
    sentiment, confidence = prediction["sentiment"], prediction["confidence"]
    return jsonify({
        "text": text,
        "sentiment": sentiment,
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    app.run(debug=True)
