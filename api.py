import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

model_name = "rafalposwiata/deproberta-large-depression"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def classify_depression(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities).item()

    return {
        "label": model.config.id2label[predicted_class],
        "confidence": float(probabilities[0][predicted_class])
    }


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = classify_depression(text)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
