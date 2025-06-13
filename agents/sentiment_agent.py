from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig
import torch
import numpy as np
import os

model_path = os.path.join("models", "finbert")

# Explicitly load config with correct number of labels (3)
config = BertConfig.from_pretrained(model_path, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)

def analyze_sentiment(news_headlines):
    if not news_headlines:
        return "NEUTRAL"

    scores = []

    for text in news_headlines:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]
        sentiment_score = probs[2] - probs[0]  # positive - negative
        scores.append(sentiment_score)

    avg_score = np.mean(scores)

    if avg_score > 0.2:
        return "POSITIVE_SENTIMENT"
    elif avg_score < -0.2:
        return "NEGATIVE_SENTIMENT"
    else:
        return "NEUTRAL"


