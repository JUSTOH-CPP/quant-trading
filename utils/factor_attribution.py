import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def factor_regression(strategy_returns, ff3, name):
    """
    Run Fama-French 3-factor regression on strategy returns.
    Returns alpha, betas, R-squared, t-statistics.
    """
    # Align all series
    aligned = pd.concat([strategy_returns, ff3], axis=1).dropna()
    aligned.columns = ['ret', 'Mkt-RF', 'SMB', 'HML', 'RF']

    # Excess strategy return over risk-free rate
    y = aligned['ret'] - aligned['RF']

    # Factor matrix
    X = aligned[['Mkt-RF', 'SMB', 'HML']]
    X = sm.add_constant(X)

    # OLS regression
    model = sm.OLS(y, X).fit()

    alpha_daily = model.params['const']
    alpha_ann   = alpha_daily * 252

    print(f"\n{'='*55}")
    print(f"  FACTOR ATTRIBUTION — {name}")
    print(f"{'='*55}")
    print(f"  Alpha (daily)    : {alpha_daily:.6f}")
    print(f"  Alpha (annual)   : {alpha_ann:.2%}")
    print(f"  Alpha t-stat     : {model.tvalues['const']:.2f}")
    print(f"  Alpha p-value    : {model.pvalues['const']:.4f}")
    print(f"  Mkt-RF beta      : {model.params['Mkt-RF']:.3f} (t={model.tvalues['Mkt-RF']:.2f})")
    print(f"  SMB beta         : {model.params['SMB']:.3f} (t={model.tvalues['SMB']:.2f})")
    print(f"  HML beta         : {model.params['HML']:.3f} (t={model.tvalues['HML']:.2f})")
    print(f"  R-squared        : {model.rsquared:.3f}")
    print(f"{'='*55}")
    print(f"  Significant alpha? {'YES' if abs(model.tvalues['const']) > 2 else 'NO'} (need |t| > 2)")

    return model


def rolling_alpha(strategy_returns, ff3, name, window=252):
    """
    Calculate rolling 1-year alpha to see
    if the edge is consistent over time.
    """
    aligned = pd.concat([strategy_returns, ff3],
                        axis=1, sort=True).dropna()
    aligned.columns = ['ret', 'Mkt-RF', 'SMB', 'HML', 'RF']

    y = aligned['ret'] - aligned['RF']
    X = sm.add_constant(aligned[['Mkt-RF', 'SMB', 'HML']])

    alphas = []
    dates  = []

    for i in range(window, len(y)):
        y_w = y.iloc[i-window:i]
        X_w = X.iloc[i-window:i]
        try:
            m = sm.OLS(y_w, X_w).fit()
            alphas.append(m.params['const'] * 252)
            dates.append(y.index[i])
        except:
            continue

    alpha_series = pd.Series(alphas, index=dates)
    return alpha_series
