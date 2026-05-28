import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, jarque_bera, probplot

def distribution_analysis(ticker, start="2022-01-01", end="2024-01-01"):
    """
    Full statistical distribution analysis for any ticker.
    Shows 3-panel diagnostic chart and prints stats summary.
    """
    # --- Data ---
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    r = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    mu, sigma = r.mean(), r.std()

    # --- Stats ---
    _, p_sw = shapiro(r)
    _, p_jb = jarque_bera(r)
    tail_actual = (np.abs(r) > 3*sigma).sum()
    tail_normal = 2 * (1 - norm.cdf(3)) * len(r)

    # --- Print ---
    print(f"\n{'='*45}")
    print(f"  {ticker} DISTRIBUTION ANALYSIS")
    print(f"{'='*45}")
    print(f"  Skewness         : {r.skew():>8.4f}")
    print(f"  Excess Kurtosis  : {r.kurt():>8.4f}")
    print(f"  Shapiro p-value  : {p_sw:>8.6f}")
    print(f"  Jarque-Bera p    : {p_jb:>8.6f}")
    print(f"  Tail events (3σ) : {tail_actual} actual vs {tail_normal:.1f} normal")
    print(f"  Fat tail ratio   : {tail_actual/tail_normal:.1f}x more extreme than normal")
    print(f"  Normal dist?     : {'NO' if p_sw < 0.05 else 'YES'}")

    # --- 3-panel chart ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: histogram vs normal
    x = np.linspace(r.min(), r.max(), 300)
    ax1.hist(r, bins=70, density=True, color='#2563eb',
             alpha=0.6, label='Actual')
    ax1.plot(x, norm.pdf(x, mu, sigma), '#ef4444', lw=2, label='Normal')
    ax1.set_title(f'{ticker} vs Normal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: QQ plot
    probplot(r, dist="norm", plot=ax2)
    ax2.set_title("QQ Plot")
    ax2.get_lines()[0].set(color='#2563eb', markersize=3)
    ax2.get_lines()[1].set(color='#ef4444', lw=2)
    ax2.grid(True, alpha=0.3)

    # Panel 3: rolling volatility
    rolling_vol = r.rolling(30).std() * np.sqrt(252)
    ax3.plot(rolling_vol, color='#7c3aed', lw=1.2)
    ax3.axhline(sigma * np.sqrt(252), color='#ef4444',
                lw=1.5, linestyle='--', label='Avg Vol')
    ax3.set_title("30-Day Rolling Volatility")
    ax3.set_ylabel("Ann. Volatility")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle(f"{ticker} Distribution Analysis", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()
