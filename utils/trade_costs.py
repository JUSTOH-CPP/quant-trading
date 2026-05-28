import numpy as np
import pandas as pd 
import yfinance as yf

def estimate_spread(ticker, start="2023-01-01", end="2024-01-01"):
    """Estimate bid-ask spread using high-low proxy"""
    df=yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.columns=df.columns.get_level_values(0)

    hl_ratio =np.log(df["High"] / df["Low"])
    spread_est =2 * (np.exp(hl_ratio /2) -1) / (1 + np.exp(hl_ratio / 2))

    avg_spread = spread_est.mean()
    median_spread = spread_est.median()
    avg_price =df["Close"].mean()

    print(f"\n{ticker} SPREAD ESTIMATE")
    print(f" Avg spread (%) : {avg_spread:.4%}")
    print(f" Median spread (%) : {median_spread:.4%}")
    print(f" Avg price : ${avg_price:.2f}")
    print(f" Spread in dollars : ${avg_price * avg_spread:.4f}")
    return avg_spread


def trade_cost_calculator(
    position_value,
    spread_pct   = 0.05,
    slippage_pct = 0.02,
    commission   = 0.0,
    n_trades_yr  = 50
):
    """
    Calculate round-trip transaction cost for a position.

    Parameters:
    -----------
    position_value : float — dollar value of position
    spread_pct     : float — one-way spread as % of price
    slippage_pct   : float — one-way slippage as % of price
    commission     : float — flat commission per trade ($)
    n_trades_yr    : int   — number of round trips per year
    """
    # One-way costs
    spread_cost   = position_value * (spread_pct   / 100)
    slippage_cost = position_value * (slippage_pct / 100)
    one_way_cost  = spread_cost + slippage_cost + commission

    # Round-trip
    roundtrip_cost = one_way_cost * 2
    roundtrip_pct  = roundtrip_cost / position_value * 100

    # Annual drag
    annual_cost = roundtrip_cost * n_trades_yr
    annual_drag = annual_cost / position_value * 100

    print(f"\n{'='*45}")
    print(f"  TRADE COST ANALYSIS")
    print(f"{'='*45}")
    print(f"  Position size      : ${position_value:>10,.0f}")
    print(f"  Spread cost        : ${spread_cost:>10.2f}  ({spread_pct:.3f}% one-way)")
    print(f"  Slippage cost      : ${slippage_cost:>10.2f}  ({slippage_pct:.3f}% one-way)")
    print(f"  Commission         : ${commission:>10.2f}  per side")
    print(f"  Round-trip cost    : ${roundtrip_cost:>10.2f}  ({roundtrip_pct:.3f}%)")
    print(f"  Trades per year    : {n_trades_yr:>10}")
    print(f"  Annual cost drag   : ${annual_cost:>10.2f}  ({annual_drag:.2f}% of capital)")
    print(f"{'='*45}")
    print(f"  Strategy must earn > {annual_drag:.2f}% just to break even on costs")

    return {"roundtrip_pct": roundtrip_pct, "annual_drag": annual_drag}
