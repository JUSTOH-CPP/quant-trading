import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf

def adf_test(series, name):
    """
    Run Augmented Dickey-Fuller stationarity test.
    H0: series has a unit root (non-stationary)
    If p<0.05: reject H0, series IS stationary
    """
    result = adfuller(series.dropna())

    print(f"\n{name} - ADF Test")
    print(f"{'='*40}")
    print(f" ADF Statistic : {result[0]:>10.4f}")
    print(f" p-value : {result[1]:>10.6f}")
    print(f" Critical 1% : {result[4]['1%']:>10.4f}")
    print(f" Critical 5% : {result[4]['5%']:>10.4f}")
    print(f"{'='*40}")
    print(f" Result: {'STATIONARY' if result[1]<0.05 else 'NON-STATIONARY'}")


def timeseries_analysis(ticker, start="2018-01-01", end="2024-01-01"):
    """
    Full time series diagnostic for any ticker.
    Tests stationarity, autocorrelation, and rolling stats.
    """
    # --- Data ---
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    price = df["Close"]
    returns = np.log(price / price.shift(1)).dropna()

    # --- ADF Tests ---
    adf_price   = adfuller(price)
    adf_returns = adfuller(returns)

    print(f"\n{'='*45}")
    print(f"  {ticker} TIME SERIES ANALYSIS")
    print(f"{'='*45}")
    print(f"  Price stationary?   : {'YES' if adf_price[1]   < 0.05 else 'NO'} (p={adf_price[1]:.4f})")
    print(f"  Returns stationary? : {'YES' if adf_returns[1] < 0.05 else 'NO'} (p={adf_returns[1]:.4f})")

    # --- Autocorrelation summary ---
    acf_vals = acf(returns, nlags=10, fft=True)
    sig_lags = [i for i, v in enumerate(acf_vals[1:], 1)
                if abs(v) > 2/np.sqrt(len(returns))]
    print(f"  Significant ACF lags: {sig_lags if sig_lags else 'None — no autocorrelation'}")

    # --- Volatility clustering ---
    acf_sq = acf(returns**2, nlags=10, fft=True)
    sig_sq = [i for i, v in enumerate(acf_sq[1:], 1)
              if abs(v) > 2/np.sqrt(len(returns))]
    print(f"  Vol clustering lags : {sig_sq if sig_sq else 'None'}")
    print(f"{'='*45}")

    # --- 4-panel chart ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    # Price
    axes[0,0].plot(price, color='#2563eb', lw=1)
    axes[0,0].set_title(f'{ticker} Price')
    axes[0,0].grid(True, alpha=0.3)

    # Returns
    axes[0,1].plot(returns, color='#1D9E75', lw=0.8, alpha=0.8)
    axes[0,1].axhline(0, color='#ef4444', lw=1, linestyle='--')
    axes[0,1].set_title('Log Returns')
    axes[0,1].grid(True, alpha=0.3)

    # ACF of returns
    plot_acf(returns, lags=30, ax=axes[1,0],
             color='#2563eb', title='ACF Returns')
    axes[1,0].grid(True, alpha=0.3)

    # Rolling volatility
    roll_vol = returns.rolling(20).std() * np.sqrt(252)
    axes[1,1].plot(roll_vol, color='#7c3aed', lw=1.2)
    axes[1,1].axhline(roll_vol.mean(), color='#ef4444',
                      lw=1.5, linestyle='--', label='Avg')
    axes[1,1].set_title('20-Day Rolling Volatility')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.suptitle(f'{ticker} Time Series Diagnostics', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()
