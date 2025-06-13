from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Save locally in ./models/finbert
AutoTokenizer.from_pretrained("ProsusAI/finbert", cache_dir="models/finbert")
AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", cache_dir="models/finbert")

print("✅ FinBERT downloaded and saved to models/finbert")

