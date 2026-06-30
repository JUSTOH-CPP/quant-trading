import numpy as np
import pandas as pd
import yfinance as yf

def detect_earnings_gaps(price_series, gap_threshold=0.04):
    """
    Detect large overnight gaps as proxy for earnings events.
    NOTE: This conflates earnings reactions with general
    volatility. For genuine PEAD research, use actual
    EPS surprise data (Compustat, Zacks, IBES).
    """
    overnight_ret = price_series.pct_change()
    gaps = overnight_ret[abs(overnight_ret) > gap_threshold]
    return pd.DataFrame({
        "date": gaps.index,
        "gap_return": gaps.values,
        "type": ["BEAT" if g > 0 else "MISS"
                for g in gaps.values]
    })


def measure_drift(price_series, events_df, ticker,
                   holding_days=20):
    """Measure cumulative return path after each event."""
    p = price_series.dropna()
    log_ret = np.log(p / p.shift(1))
    ticker_events = events_df[events_df["ticker"]==ticker]
    paths = []
    for _, event in ticker_events.iterrows():
        if event["date"] not in p.index:
            continue
        idx = p.index.get_loc(event["date"])
        if idx + holding_days >= len(p):
            continue
        path = log_ret.iloc[idx+1:idx+1+holding_days].cumsum()
        paths.append({"type": event["type"],
                      "path": path.values})
    return paths


def fetch_fred_series(series_id, start="2010-01-01"):
    """
    Fetch a FRED economic series via public CSV API.
    No API key required. Useful series:
    T10Y2Y = yield curve (recession predictor)
    UNRATE = unemployment rate
    CPIAUCSL = inflation (CPI)
    VIXCLS = VIX from FRED (alternative source)
    """
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv"
           f"?id={series_id}&cosd={start}")
    df = pd.read_csv(url)
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df[series_id] = pd.to_numeric(df[series_id],
                                   errors="coerce")
    return df.dropna()
