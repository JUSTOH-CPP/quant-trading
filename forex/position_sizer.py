import numpy as np
import pandas as pd

def position_sizer(account_balance, risk_pct,
                   entry, sl, pair="XAUUSD"):
    """
    Calculate position size for forex/gold trades.
    Returns lots, mini lots, and micro lots.
    """
    risk_usd  = account_balance * risk_pct
    stop_pips = abs(entry - sl)

    if pair == "XAUUSD":
        pip_value_per_lot = 100
    elif "JPY" in pair:
        pip_value_per_lot = 1000
    else:
        pip_value_per_lot = 100000 * 0.0001 * 10

    lots = risk_usd / (stop_pips * pip_value_per_lot)

    return {
        "risk_usd"  : round(risk_usd, 2),
        "stop_pips" : round(stop_pips, 2),
        "lots"      : round(lots, 3),
        "mini_lots" : round(lots * 10, 2),
        "micro_lots": round(lots * 100, 1)
    }


def kelly_position_size(win_rate, avg_win_r, avg_loss_r,
                         account, max_risk=0.02):
    """
    Kelly-based position sizing.
    Caps at max_risk for safety.
    Returns recommended risk % and dollar risk.
    """
    kelly    = win_rate - ((1-win_rate) / (avg_win_r/avg_loss_r))
    half_k   = kelly / 2
    safe_pct = min(half_k, max_risk)

    print(f"Full Kelly    : {kelly:.1%}")
    print(f"Half Kelly    : {half_k:.1%}")
    print(f"Safe risk %   : {safe_pct:.1%}")
    print(f"Safe risk $   : ${account * safe_pct:.0f}")
    return safe_pct
