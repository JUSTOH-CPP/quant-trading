import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

class PaperTrader:
    """
    Paper trading engine.
    Tracks positions, cash, trades and equity curve.
    Includes drawdown kill switch.
    """
    def __init__(self, capital=100000,
                 max_drawdown_pct=0.15):
        self.initial_capital  = capital
        self.cash             = capital
        self.positions        = {}
        self.trades           = []
        self.equity_curve     = []
        self.max_drawdown_pct = max_drawdown_pct
        self.peak_equity      = capital
        self.killed           = False

    def get_portfolio_value(self, current_prices):
        holdings = sum(self.positions.get(t, 0) *
                       current_prices.get(t, 0)
                       for t in self.positions)
        return self.cash + holdings

    def execute(self, ticker, target_shares,
                price, date):
        if self.killed:
            return
        current_shares = self.positions.get(ticker, 0)
        order_shares   = target_shares - current_shares
        if order_shares == 0:
            return
        cost     = order_shares * price
        cost_pct = abs(order_shares) * price * 0.0006
        if order_shares > 0 and cost+cost_pct > self.cash:
            affordable   = int(self.cash / (price*1.0006))
            order_shares = max(0, affordable)
            cost         = order_shares * price
            cost_pct     = order_shares * price * 0.0006
        if order_shares == 0:
            return
        self.cash -= (cost + cost_pct)
        self.positions[ticker] = current_shares + order_shares
        if self.positions[ticker] == 0:
            del self.positions[ticker]
        self.trades.append({
            "date": date, "ticker": ticker,
            "shares": order_shares, "price": price,
            "action": "BUY" if order_shares>0 else "SELL"
        })

    def mark_to_market(self, date, current_prices):
        val = self.get_portfolio_value(current_prices)
        self.equity_curve.append({"date": date,
                                   "equity": val})
        if val > self.peak_equity:
            self.peak_equity = val
        dd = (val - self.peak_equity) / self.peak_equity
        if dd < -self.max_drawdown_pct and not self.killed:
            self.killed = True
        return val, dd

    def report(self):
        eq    = pd.DataFrame(self.equity_curve)
        final = eq["equity"].iloc[-1] if len(eq) else self.cash
        ret   = (final / self.initial_capital) - 1
        print(f"Return: {ret:.2%}  Trades: {len(self.trades)}"
              f"  Kill switch: {'ON' if self.killed else 'OFF'}")
