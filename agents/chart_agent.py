import yfinance as yf

def analyze_chart(ticker="AAPL", period="7d", interval="1h"):
    data = yf.download(ticker, period=period, interval=interval)

    if data.empty:
        return "NO_DATA"

    close = data["Close"].dropna()
    ma_fast = close.rolling(window=5).mean()
    ma_slow = close.rolling(window=20).mean()

    if ma_fast.dropna().empty or ma_slow.dropna().empty:
        return "INSUFFICIENT_DATA"

    latest_fast = ma_fast.dropna().iloc[-1].item()
    latest_slow = ma_slow.dropna().iloc[-1].item()

    if latest_fast > latest_slow:
        return "BULLISH_CROSSOVER"
    elif latest_fast < latest_slow:
        return "BEARISH_CROSSOVER"
    else:
        return "NEUTRAL"





