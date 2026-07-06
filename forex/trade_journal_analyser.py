import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def add_trade(df, date, pair, direction, entry, sl, tp,
              exit_price, outcome, session, setup, notes=""):
    """
    Add a single trade to the journal.
    outcome: WIN, LOSS, or BE
    session: London, NY, Tokyo, Sydney
    setup  : SMC, Lipschutz, Combined
    """
    risk_pips   = abs(entry - sl)
    reward_pips = abs(tp - entry)
    rr_ratio    = reward_pips / risk_pips

    if outcome == "WIN":
        actual_r = abs(exit_price - entry) / risk_pips
    elif outcome == "LOSS":
        actual_r = -abs(exit_price - entry) / risk_pips
    else:
        actual_r = 0.0

    new_trade = pd.DataFrame([{
        "date"       : pd.to_datetime(date),
        "pair"       : pair,
        "direction"  : direction,
        "entry"      : entry,
        "sl"         : sl,
        "tp"         : tp,
        "exit_price" : exit_price,
        "outcome"    : outcome,
        "session"    : session,
        "setup"      : setup,
        "notes"      : notes,
        "risk_pips"  : risk_pips,
        "reward_pips": reward_pips,
        "rr_ratio"   : rr_ratio,
        "actual_r"   : float(actual_r)
    }])

    df = pd.concat([df, new_trade], ignore_index=True)
    print(f"  Added: {direction} {pair} {outcome} "
          f"({float(actual_r):+.2f}R)")
    return df


def analyse_journal(df):
    """Full trade journal analysis with breakdown by pair, session, setup."""
    total    = len(df)
    wins     = (df["outcome"]=="WIN").sum()
    losses   = (df["outcome"]=="LOSS").sum()
    win_rate = wins / total

    avg_win  = df[df["outcome"]=="WIN"]["actual_r"].astype(float).mean()
    avg_loss = df[df["outcome"]=="LOSS"]["actual_r"].astype(float).mean()
    exp      = (win_rate * avg_win) + ((1-win_rate) * abs(avg_loss))

    df_s = df.sort_values("date").copy()
    df_s["cumulative_r"] = df_s["actual_r"].astype(float).cumsum()
    dd_r = (df_s["cumulative_r"] - df_s["cumulative_r"].cummax()).min()

    print(f"\n{'='*55}")
    print(f"  TRADE JOURNAL ANALYSIS")
    print(f"{'='*55}")
    print(f"  Total trades   : {total}")
    print(f"  Wins/Losses    : {wins}/{losses}")
    print(f"  Win rate       : {win_rate:.1%}")
    print(f"  Avg win (R)    : +{avg_win:.2f}R")
    print(f"  Avg loss (R)   : {avg_loss:.2f}R")
    print(f"  Expectancy     : {exp:.3f}R per trade")
    print(f"  Max DD (R)     : {dd_r:.2f}R")
    print(f"  Total R earned : {df_s['actual_r'].astype(float).sum():.2f}R")
    print(f"{'='*55}")

    for col in ["pair", "session", "setup"]:
        if col in df.columns:
            print(f"\n  BY {col.upper()}:")
            for val in df[col].unique():
                sub = df[df[col]==val]
                wr  = (sub["outcome"]=="WIN").mean()
                er  = sub["actual_r"].astype(float).mean()
                print(f"  {val:<12} n:{len(sub):>3}  "
                      f"WR:{wr:.0%}  Avg R:{er:>+.2f}")

    return df_s
