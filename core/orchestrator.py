# Main orchestrator
# core/orchestrator.py

import json
from agents.chart_agent import analyze_chart
from agents.sentiment_agent import analyze_sentiment
from agents.news_agent import fetch_news
from agents.llm_decider import decide_trade
from agents.risk_manager import apply_risk_rules
from agents.feedback_evaluator import update_strategy

CONFIG_PATH = "config/settings.json"

def load_settings():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def run():
    settings = load_settings()
    
    print("🔄 Fetching latest news...")
    news = fetch_news()

    print("📊 Analyzing chart data...")
    chart_signal = analyze_chart()

    print("🧠 Running sentiment analysis...")
    sentiment = analyze_sentiment(news)

    print("🤖 Sending to LLM for decision...")
    decision = decide_trade(chart_signal, sentiment, news)

    print("⚖️ Applying risk manager...")
    final_decision = apply_risk_rules(decision)

    print("✅ Final Trade Decision:", final_decision)

    update_strategy(final_decision)
