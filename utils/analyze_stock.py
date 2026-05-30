import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

def analyze_stock(ticker, start="2022-01-01", end="2024-12-31"):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    price   = df["Close"]
    returns = np.log(price / price.shift(1)).dropna()

    ann_return = returns.mean() * 252
    ann_vol    = returns.std()  * np.sqrt(252)
    sharpe     = ann_return / ann_vol

    stats = {
        "ticker"       : ticker,
        "ann_return"   : ann_return,
        "ann_vol"      : ann_vol,
        "sharpe"       : sharpe,
        "worst_day"    : returns.min(),
        "best_day"     : returns.max(),
        "total_return" : (price.iloc[-1] / price.iloc[0]) - 1,
    }

    print(f"\n{'='*40}")
    print(f"  {ticker} | {start} to {end}")
    print(f"{'='*40}")
    for k, v in stats.items():
        if k == "ticker":
            pass
        elif k == "sharpe":
            print(f"  {k:<20} {v:>8.2f}")
        elif isinstance(v, float):
            print(f"  {k:<20} {v:>8.2%}")

    sma50 = price.rolling(window=50).mean()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax1.plot(price, lw=1.2, color="#2563eb", label="Price")
    ax1.plot(sma50, lw=1.5, color="#f97316",
             linestyle="--", label="50-day SMA")
    ax1.legend()
    ax1.set_title(f"{ticker} -- Ann. Return {ann_return:.1%} | Sharpe {sharpe:.2f}")
    ax1.grid(True, alpha=0.3)
    ax2.bar(df.index, df["Volume"], color="#94a3b8", alpha=0.7)
    ax2.set_ylabel("Volume")
    plt.tight_layout()
    plt.show()

    return stats
