import numpy as np
import pandas as pd

def twap_schedule(total_shares, n_slices, price_series):
    """
    Time-Weighted Average Price execution schedule.
    Splits order into equal time slices.
    Returns schedule DataFrame and average execution price.
    """
    slice_size = total_shares // n_slices
    remainder  = total_shares % n_slices
    interval   = len(price_series) // n_slices
    schedule   = []

    for i in range(n_slices):
        idx        = i * interval
        shares     = slice_size + (remainder if i==0 else 0)
        exec_price = price_series.iloc[idx]
        schedule.append({
            "slice": i+1,
            "time" : price_series.index[idx],
            "shares": shares,
            "price": exec_price,
            "value": shares * exec_price
        })

    df = pd.DataFrame(schedule)
    avg_price = df["value"].sum() / df["shares"].sum()
    return df, avg_price


def vwap_schedule(total_shares, price_series, n_slices=12):
    """
    Volume-Weighted Average Price execution schedule.
    Uses U-shaped intraday volume profile.
    Returns schedule DataFrame and average execution price.
    """
    volume_profile = np.array([
        0.15, 0.10, 0.07, 0.06, 0.05,
        0.05, 0.06, 0.07, 0.08, 0.10,
        0.10, 0.11
    ])
    volume_profile = volume_profile / volume_profile.sum()
    interval = len(price_series) // n_slices
    schedule = []

    for i in range(n_slices):
        idx    = i * interval
        shares = int(total_shares * volume_profile[i])
        schedule.append({
            "slice"  : i+1,
            "time"   : price_series.index[idx],
            "pct_vol": volume_profile[i],
            "shares" : shares,
            "price"  : price_series.iloc[idx],
            "value"  : shares * price_series.iloc[idx]
        })

    df   = pd.DataFrame(schedule)
    diff = total_shares - df["shares"].sum()
    df.loc[df.index[-1], "shares"] += diff
    df["value"] = df["shares"] * df["price"]
    avg_price   = df["value"].sum() / df["shares"].sum()
    return df, avg_price


def implementation_shortfall(decision_price, exec_price,
                              total_shares):
    """
    Calculate implementation shortfall in bps and dollars.
    """
    shortfall_pct  = (exec_price - decision_price) / decision_price
    shortfall_bps  = shortfall_pct * 10000
    shortfall_usd  = (exec_price - decision_price) * total_shares
    return {
        "shortfall_pct": shortfall_pct,
        "shortfall_bps": shortfall_bps,
        "shortfall_usd": shortfall_usd
    }
