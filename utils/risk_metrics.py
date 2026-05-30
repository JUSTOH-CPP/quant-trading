import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def max_drawdown(returns):
    #Build cumulative wealth index
    wealth = (1 + returns).cumprod()

    #Rolling peak
    rolling_peak = wealth.cummax()

    #Drawdown at each point
    drawdown = (wealth - rolling_peak) / rolling_peak

    max_dd = drawdown.min()
    max_dd_date =drawdown.idxmin()

    return max_dd, max_dd_date, drawdown


def var_cvar(returns, confidence=0.95):
    """
    Historical VaR and CVaR at given confidence level.
    VaR-worst loss on (1-confidence)% of days
    CVaR-average loss on those worst days
    """
    sorted_returns = returns.sort_values()
    cutoff_idx = int((1 - confidence) * len(sorted_returns))

    var = sorted_returns.iloc[cutoff_idx]
    cvar = sorted_returns.iloc[:cutoff_idx].mean()

    return var, cvar


def risk_report(ticker, start="2019-01-01", end="2024-01-01"):
    """
    Complete risk report for any ticker.
    Sharpe, Sortino, Max Drawdown, VaR, CVaR.
    """
    # --- Data ---
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    price   = df["Close"]
    returns = np.log(price / price.shift(1)).dropna()

    # --- Metrics ---
    rf_daily      = 0.02 / 252
    excess        = returns - rf_daily
    sharpe        = excess.mean() / excess.std() * np.sqrt(252)
    down_std      = returns[returns < rf_daily].std() * np.sqrt(252)
    sortino       = (returns.mean() * 252 - 0.02) / down_std
    wealth        = (1 + returns).cumprod()
    drawdown      = (wealth - wealth.cummax()) / wealth.cummax()
    max_dd        = drawdown.min()
    max_dd_date   = drawdown.idxmin()
    var_95        = returns.sort_values().iloc[int(0.05 * len(returns))]
    cvar_95       = returns.sort_values().iloc[:int(0.05 * len(returns))].mean()
    var_99        = returns.sort_values().iloc[int(0.01 * len(returns))]
    cvar_99       = returns.sort_values().iloc[:int(0.01 * len(returns))].mean()
    ann_return    = returns.mean() * 252
    ann_vol       = returns.std()  * np.sqrt(252)

    # --- Print ---
    print(f"\n{'='*45}")
    print(f"  {ticker} RISK REPORT")
    print(f"{'='*45}")
    print(f"  Ann. Return    : {ann_return:>8.2%}")
    print(f"  Ann. Volatility: {ann_vol:>8.2%}")
    print(f"  Sharpe Ratio   : {sharpe:>8.2f}")
    print(f"  Sortino Ratio  : {sortino:>8.2f}")
    print(f"  Max Drawdown   : {max_dd:>8.2%}  ({max_dd_date.date()})")
    print(f"  95% VaR        : {var_95:>8.2%}")
    print(f"  95% CVaR       : {cvar_95:>8.2%}")
    print(f"  99% VaR        : {var_99:>8.2%}")
    print(f"  99% CVaR       : {cvar_99:>8.2%}")
    print(f"{'='*45}")

    # --- 3 panel chart ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 9),
        gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)

    ax1.plot(wealth, color='#2563eb', lw=1.2)
    ax1.set_title(f'{ticker} — Sharpe {sharpe:.2f} | Max DD {max_dd:.1%}')
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(drawdown.index, drawdown, 0,
                     color='#ef4444', alpha=0.4)
    ax2.set_title('Drawdown')
    ax2.grid(True, alpha=0.3)

    roll_vol = returns.rolling(20).std() * np.sqrt(252)
    ax3.plot(roll_vol, color='#7c3aed', lw=1)
    ax3.axhline(ann_vol, color='#ef4444', lw=1.5,
                linestyle='--', label=f'Avg {ann_vol:.1%}')
    ax3.set_title('Rolling Volatility')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {"sharpe": sharpe, "sortino": sortino,
            "max_dd": max_dd, "var_95": var_95, "cvar_99": cvar_99}
