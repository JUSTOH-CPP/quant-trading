import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── CONFIGURATION ────────────────────────────────────────
BASELINES = {
    "XAUUSD": {"wr":0.625,"exp":1.91,
               "avg_w":2.46,"avg_l":-1.00,
               "label":"Gold"},
    "XAGUSD": {"wr":0.50,"exp":1.00,
               "avg_w":2.00,"avg_l":-1.00,
               "label":"Silver"},
    "NAS100": {"wr":0.50,"exp":1.00,
               "avg_w":2.00,"avg_l":-1.00,
               "label":"Nasdaq 100"},
    "GBPUSD": {"wr":0.50,"exp":1.00,
               "avg_w":2.00,"avg_l":-1.00,
               "label":"Cable"},
    "GBPJPY": {"wr":0.50,"exp":1.00,
               "avg_w":2.00,"avg_l":-1.00,
               "label":"The Beast"},
}

RISK_PER_TRADE  = 100
ACCOUNT         = 10000
ALERT_WR        = 0.45
ALERT_EXP       = 1.00
ALERT_STREAK    = 3
ALERT_MONTHLY_R = 10.0
SETUPS          = ["SMC","Lipschutz","Powell","Combined"]


def add_trade(df, date, pair, direction, entry, sl, tp,
              exit_price, outcome, session, setup,
              notes=""):
    risk_pips   = abs(entry - sl)
    reward_pips = abs(tp - entry)
    rr_ratio    = reward_pips / risk_pips                   if risk_pips > 0 else 0
    if outcome == "WIN":
        actual_r = abs(exit_price - entry) / risk_pips
    elif outcome == "LOSS":
        actual_r = -abs(exit_price - entry) / risk_pips
    else:
        actual_r = 0.0
    pnl_usd = actual_r * RISK_PER_TRADE
    new_trade = pd.DataFrame([{
        "date"      : pd.to_datetime(date),
        "pair"      : pair,
        "direction" : direction,
        "entry"     : entry,
        "sl"        : sl,
        "tp"        : tp,
        "exit_price": exit_price,
        "outcome"   : outcome,
        "session"   : session,
        "setup"     : setup,
        "notes"     : notes,
        "risk_pips" : round(risk_pips, 2),
        "rr_ratio"  : round(rr_ratio, 2),
        "actual_r"  : round(float(actual_r), 3),
        "pnl_usd"   : round(float(pnl_usd), 2)
    }])
    df = pd.concat([df, new_trade], ignore_index=True)
    print(f"  + {date} | {pair} {direction} "
          f"{outcome} {float(actual_r):+.2f}R "
          f"(${float(pnl_usd):+.2f})")
    return df


def load_journal(path="live_journal.csv"):
    try:
        df = pd.read_csv(path,
                         parse_dates=["date"])
        print(f"Loaded {len(df)} trades from {path}")
        return df
    except FileNotFoundError:
        print("No existing journal found. Starting fresh.")
        return pd.DataFrame(columns=[
            "date","pair","direction","entry","sl","tp",
            "exit_price","outcome","session","setup",
            "notes","risk_pips","rr_ratio",
            "actual_r","pnl_usd"])


def save_journal(df, path="live_journal.csv"):
    df.to_csv(path, index=False)
    print(f"Journal saved: {len(df)} trades → {path}")


def analyse_journal(df, pair=None):
    if pair:
        df = df[df["pair"]==pair].copy()
    if len(df) == 0:
        return None
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    total  = len(df)
    wins   = (df["outcome"]=="WIN").sum()
    losses = (df["outcome"]=="LOSS").sum()
    be     = (df["outcome"]=="BE").sum()
    wr     = wins / total
    avg_w  = df[df["outcome"]=="WIN"]["actual_r"].mean()
    avg_l  = df[df["outcome"]=="LOSS"]["actual_r"].mean()
    exp    = (wr*avg_w)+((1-wr)*avg_l)              if not pd.isna(avg_l) else wr*avg_w
    tot_r  = df["actual_r"].sum()
    tot_usd= df["pnl_usd"].sum()
    recent     = df.tail(10)
    recent_wr  = (recent["outcome"]=="WIN").mean()
    recent_exp = recent["actual_r"].mean()
    streak = 0; streak_type = None
    for _, row in df.iloc[::-1].iterrows():
        if streak == 0:
            streak_type = row["outcome"]; streak = 1
        elif row["outcome"] == streak_type:
            streak += 1
        else:
            break
    df["month"] = df["date"].dt.strftime("%Y-%m")
    monthly = df.groupby("month").agg(
        trades  = ("actual_r","count"),
        total_r = ("actual_r","sum"),
        wr      = ("outcome",
                   lambda x: (x=="WIN").mean())
    ).round(3)
    return {
        "total":total,"wins":wins,"losses":losses,
        "be":be,"wr":wr,"avg_w":avg_w,"avg_l":avg_l,
        "exp":exp,"tot_r":tot_r,"tot_usd":tot_usd,
        "recent_wr":recent_wr,"recent_exp":recent_exp,
        "streak":streak,"streak_type":streak_type,
        "monthly":monthly,"df":df
    }


def check_alerts(stats, pair, baseline):
    alerts = []
    if stats["recent_wr"] < ALERT_WR:
        alerts.append(
            f"WR ALERT: Last 10 WR "
            f"{stats['recent_wr']:.0%} < {ALERT_WR:.0%}")
    if stats["recent_exp"] < ALERT_EXP:
        alerts.append(
            f"EXP ALERT: Last 10 exp "
            f"{stats['recent_exp']:+.2f}R < +{ALERT_EXP:.2f}R")
    if stats["streak_type"] == "LOSS" and        stats["streak"] >= ALERT_STREAK:
        alerts.append(
            f"STREAK: {stats['streak']} losses "
            f"— reduce size to 0.5%")
    if stats["wr"]  - baseline["wr"]  < -0.15:
        alerts.append("DRIFT: WR drifted >15% below baseline")
    if stats["exp"] - baseline["exp"] < -0.75:
        alerts.append("DRIFT: Exp drifted >0.75R below baseline")
    return alerts


def dynamic_position_size(df, pair,
                           account=ACCOUNT):
    stats = analyse_journal(df, pair)
    if stats is None or stats["total"] < 10:
        baseline = BASELINES.get(pair,
            {"wr":0.5,"avg_w":2.0,"avg_l":-1.0})
        wr=baseline["wr"]; avg_w=baseline["avg_w"]
        avg_l=abs(baseline["avg_l"]); source="BASELINE"
    else:
        wr=stats["wr"]; avg_w=stats["avg_w"]
        avg_l=abs(stats["avg_l"]); source="LIVE"
    kelly    = wr - ((1-wr)/(avg_w/avg_l))
    safe_pct = min(max(kelly/2, 0.005), 0.02)
    return {"pair":pair,"source":source,"wr":wr,
            "kelly":kelly,"safe_pct":safe_pct,
            "risk_usd":account*safe_pct}
