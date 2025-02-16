import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
        "confidence": probabilities[0][predicted_class].item(),
    }


result = classify_depression(
    "I'll kill myself, I'm so depressed and alone, Thinking to suicide")
print(
    f"Classification: {result['label']} (confidence: {result['confidence']:.2%})")
