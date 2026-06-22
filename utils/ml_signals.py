import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys
sys.path.append("../utils")
from indicators import compute_rsi

def build_features(price, high, low, volume, vix, log_returns):
    """
    Build ML feature matrix from technical indicators.
    All features use only data available at close T.
    Target: next-day return direction (shift -1).
    No lookahead bias.
    """
    feat = pd.DataFrame(index=price.index)

    feat["ret_1d"]  = price.pct_change(1)
    feat["ret_5d"]  = price.pct_change(5)
    feat["ret_10d"] = price.pct_change(10)
    feat["ret_20d"] = price.pct_change(20)
    feat["ret_60d"] = price.pct_change(60)

    feat["rsi2"]  = compute_rsi(price, period=2)
    feat["rsi14"] = compute_rsi(price, period=14)

    sma20  = price.rolling(20).mean()
    sma50  = price.rolling(50).mean()
    sma200 = price.rolling(200).mean()
    feat["price_vs_sma20"]  = (price - sma20)  / sma20
    feat["price_vs_sma50"]  = (price - sma50)  / sma50
    feat["price_vs_sma200"] = (price - sma200) / sma200

    feat["vol_20d"]    = log_returns.rolling(20).std() * np.sqrt(252)
    feat["vix"]        = vix
    feat["vix_regime"] = (vix > 20).astype(int)
    feat["bull_regime"]= (price > sma200).astype(int)

    prev_close = price.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low  - prev_close).abs()], axis=1).max(axis=1)
    feat["atr_pct"]   = tr.rolling(14).mean() / price
    feat["zscore_20"] = (price - sma20) / price.rolling(20).std()
    feat["vol_ratio"] = volume / volume.rolling(20).mean()

    feat["target"] = (log_returns.shift(-1) > 0).astype(int)
    return feat.dropna()


def train_rf_signal(X_train, y_train,
                    n_estimators=300, max_depth=3,
                    min_samples_leaf=80):
    """Train Random Forest signal classifier."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_signal(rf, X_train, y_train, X_test, y_test):
    """Print IS vs OOS accuracy and gap."""
    is_acc  = rf.score(X_train, y_train)
    oos_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"IS accuracy : {is_acc:.3f}")
    print(f"OOS accuracy: {oos_acc:.3f}")
    print(f"Gap         : {is_acc-oos_acc:.3f}")
    return is_acc, oos_acc
