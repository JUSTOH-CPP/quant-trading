import numpy as np
import pandas as pd
from collections import defaultdict


class LimitOrderBook:
    """
    Simplified limit order book simulator.
    Accepts limit and market orders, matches them,
    and tracks trade history and mid-price.
    """
    def __init__(self):
        self.bids   = defaultdict(int)
        self.asks   = defaultdict(int)
        self.trades = []
        self.time   = 0

    def best_bid(self):
        return max(self.bids.keys()) if self.bids else None

    def best_ask(self):
        return min(self.asks.keys()) if self.asks else None

    def mid_price(self):
        bb, ba = self.best_bid(), self.best_ask()
        return (bb + ba) / 2 if bb and ba else None

    def spread(self):
        bb, ba = self.best_bid(), self.best_ask()
        return ba - bb if bb and ba else None

    def add_limit_order(self, side, price, shares):
        price = round(price, 2)
        if side == "buy":
            self.bids[price] += shares
        else:
            self.asks[price] += shares
        self._match()

    def add_market_order(self, side, shares):
        fills = []
        book  = self.asks if side=="buy" else self.bids
        get_best = self.best_ask if side=="buy" else self.best_bid
        while shares > 0 and book:
            best      = get_best()
            fill      = min(shares, book[best])
            book[best] -= fill
            if book[best] == 0:
                del book[best]
            fills.append((best, fill))
            self.trades.append({"time": self.time,
                                 "price": best,
                                 "shares": fill,
                                 "side": side})
            shares -= fill
        self.time += 1
        return fills

    def _match(self):
        while self.bids and self.asks:
            bb, ba = self.best_bid(), self.best_ask()
            if bb >= ba:
                fill = min(self.bids[bb], self.asks[ba])
                self.bids[bb] -= fill
                self.asks[ba] -= fill
                if self.bids[bb] == 0: del self.bids[bb]
                if self.asks[ba] == 0: del self.asks[ba]
                self.trades.append({"time": self.time,
                                     "price": ba,
                                     "shares": fill,
                                     "side": "match"})
                self.time += 1
            else:
                break


def rolls_spread(prices):
    """
    Roll (1984) bid-ask spread estimator from daily prices.
    spread = 2 * sqrt(-cov(dP_t, dP_{t-1}))
    """
    dp  = prices.diff().dropna()
    cov = dp.cov(dp.shift(1).dropna().reindex(dp.index))
    return 2 * np.sqrt(-cov) if cov < 0 else 0


def market_impact(order_size_usd, adv_usd,
                   daily_vol, sigma=0.1):
    """
    Square-root market impact model.
    impact = sigma * daily_vol * sqrt(order/ADV)
    Returns impact as fraction of price.
    """
    participation = order_size_usd / adv_usd
    return sigma * daily_vol * np.sqrt(participation)
