import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def walk_forward(price, strategy_fn,
                 train_years=3, test_years=1,
                 cost_pct=0.0006):
    """
    Walk-forward test any strategy function.
    strategy_fn(train_price, test_price) -> position Series
    """
    train_days  = train_years * 252
    test_days   = test_years  * 252
    oos_returns = []
    windows     = []

    for start in range(0, len(price)-train_days-test_days,
                       test_days):
        train = price.iloc[start : start+train_days]
        test  = price.iloc[start+train_days :
                           start+train_days+test_days]
        pos      = strategy_fn(train, test)
        pos      = pos.reindex(test.index).fillna(0)
        test_ret = np.log(test / test.shift(1)).dropna()
        trades   = pos.diff().abs().fillna(0)
        net_ret  = pos.reindex(test_ret.index).fillna(0) * test_ret
        net_ret  = net_ret - trades.reindex(
                       test_ret.index).fillna(0) * cost_pct
        oos_returns.append(net_ret)
        windows.append({
            "train_start": train.index[0].date(),
            "train_end"  : train.index[-1].date(),
            "test_start" : test.index[0].date(),
            "test_end"   : test.index[-1].date(),
        })

    return pd.concat(oos_returns), windows


def monte_carlo_test_v2(position, returns, n_sims=1000,
                         cost_pct=0.0006):
    """
    Monte Carlo significance test via position shuffling.
    """
    def strategy_sharpe(pos, ret):
        trades = pos.diff().abs().fillna(0)
        net    = pos * ret - trades * cost_pct
        ar = net.mean() * 252
        av = net.std()  * np.sqrt(252)
        return (ar - 0.02) / av if av > 0 else 0

    pos            = position.reindex(returns.index).fillna(0)
    real_sharpe    = strategy_sharpe(pos, returns)
    random_sharpes = []

    for _ in range(n_sims):
        shuffled = pd.Series(
            np.random.permutation(pos.values),
            index=returns.index)
        random_sharpes.append(strategy_sharpe(shuffled, returns))

    random_sharpes = np.array(random_sharpes)
    pct_beat = np.mean(random_sharpes < real_sharpe)

    print(f"Strategy Sharpe : {real_sharpe:.4f}")
    print(f"Random mean     : {random_sharpes.mean():.4f}")
    print(f"Beats random    : {pct_beat:.1%}")
    print(f"{'SIGNIFICANT' if pct_beat > 0.95 else 'NOT SIGNIFICANT'} at 95%")

    return pct_beat
