Quant Trading Research — JUSTOH-CPP



Active forex trader and quantitative researcher | XAU/USD · XAG/USD · NAS100 · GBP/USD · GBP/JPY



This repository documents a complete quantitative trading research programme — from foundational statistics through live strategy validation — built over 30 days and extended with applied forex research. It combines systematic backtesting, rigorous anti-overfitting validation, and real-world live trade analysis across multiple instruments.



Key Findings

Finding	Result

Live trading edge (16 real XAU/USD trades)	+1.91R expectancy per trade

SMC mechanical alone	-0.07R (loses without confirmation)

LIP-SIM mechanical alone	-0.02R (breakeven without context)

SMC + LIP-SIM combined	+0.30R mechanical edge

Powell 10AM mechanical (SHORT + Manip>0.1%)	+0.333R expectancy

Final programme portfolio Sharpe	0.49 vs buy \& hold 0.45

Final portfolio max drawdown	-15.41% vs buy \& hold -35.75%



The fusion of SMC and LIP-SIM creates genuine mechanical edge where neither system alone has any. The discretionary SMC overlay adds approximately +1.6R per trade on top of the mechanical baseline — quantifying the skill premium for the first time.



Repository Structure

quant-trading/

│

├── week1/                      # Foundations

│   ├── day01\_stock\_analysis.ipynb

│   ├── day02\_distributions.ipynb

│   ├── day03\_trade\_costs.ipynb

│   ├── day04\_timeseries.ipynb

│   ├── day05\_risk\_metrics.ipynb

│   ├── day06\_indicators.ipynb

│   └── day07\_momentum\_screener.ipynb

│

├── week2/                      # Strategy Development

│   ├── day08\_sma\_backtest.ipynb

│   ├── day09\_mean\_reversion.ipynb

│   ├── day10\_pairs\_trading.ipynb

│   ├── day11\_rsi2\_strategy.ipynb

│   ├── day12\_volatility\_strategies.ipynb

│   ├── day13\_multi\_signal.ipynb

│   └── day14\_portfolio\_basics.ipynb

│

├── week3/                      # Validation \& Risk

│   ├── day15\_backtest\_framework.ipynb

│   ├── day16\_overfitting\_tests.ipynb

│   ├── day17\_kelly\_sizing.ipynb

│   ├── day18\_trade\_management.ipynb

│   ├── day19\_portfolio\_optimizer.ipynb

│   ├── day20\_factor\_attribution.ipynb

│   └── day21\_strategy\_audit.ipynb

│

├── week4/                      # Advanced Topics

│   ├── day22\_ml\_signals.ipynb

│   ├── day23\_execution\_algos.ipynb

│   ├── day24\_crypto\_strategies.ipynb

│   ├── day25\_paper\_trader.ipynb

│   ├── day26\_market\_microstructure.ipynb

│   ├── day27\_risk\_systems.ipynb

│   ├── day28\_alt\_data.ipynb

│   ├── day29\_final\_portfolio.ipynb

│   └── day30\_capstone.ipynb

│

├── forex/                      # Applied Forex Research

│   ├── trade\_journal.ipynb             # Live trade tracking

│   ├── lipsim\_backtest.ipynb           # LIP-SIM on GLD H1

│   ├── lipsim\_xauusd\_backtest.ipynb    # LIP-SIM on XAU/USD H1

│   ├── smc\_detector.ipynb              # Full SMC component library

│   ├── powell\_xauusd\_5min.ipynb        # Powell 10AM on XAU/USD

│   ├── combined\_backtest.ipynb         # MTF combined system

│   ├── live\_tracker.ipynb              # Live multi-pair dashboard

│   ├── live\_journal.csv                # Persistent trade log

│   ├── trade\_journal\_analyser.py       # R calculation engine

│   ├── position\_sizer.py               # Kelly position calculator

│   └── live\_tracker.py                 # Multi-pair tracker engine

│

└── utils/                      # 27-Tool Python Library

&#x20;   ├── analyze\_stock.py

&#x20;   ├── distribution\_analysis.py

&#x20;   ├── trade\_costs.py

&#x20;   ├── timeseries.py

&#x20;   ├── risk\_metrics.py

&#x20;   ├── indicators.py

&#x20;   ├── momentum\_screener.py

&#x20;   ├── sma\_backtest.py

&#x20;   ├── mean\_reversion.py

&#x20;   ├── pairs\_trading.py

&#x20;   ├── rsi2\_strategy.py

&#x20;   ├── volatility\_strategies.py

&#x20;   ├── multi\_signal.py

&#x20;   ├── backtest\_framework.py

&#x20;   ├── overfitting\_tests.py

&#x20;   ├── kelly\_sizing.py

&#x20;   ├── trade\_management.py

&#x20;   ├── portfolio\_optimizer.py

&#x20;   ├── factor\_attribution.py

&#x20;   ├── ml\_signals.py

&#x20;   ├── execution\_algos.py

&#x20;   ├── crypto\_strategies.py

&#x20;   ├── paper\_trader.py

&#x20;   ├── market\_microstructure.py

&#x20;   ├── risk\_systems.py

&#x20;   ├── alt\_data.py

&#x20;   └── final\_portfolio.py

The 30-Day Programme



A complete self-directed quantitative trading curriculum covering four weeks of increasingly advanced material.



Week 1 — Foundations



Statistical analysis of financial data, return distributions, transaction cost modelling, time series analysis, risk metrics, technical indicators, and cross-sectional momentum screening.



Week 2 — Strategy Development



SMA crossover systems, mean reversion z-score strategies, pairs trading with cointegration testing, RSI(2) with trend filters, volatility targeting, multi-signal combination, and portfolio basics.



Week 3 — Validation and Risk



Walk-forward backtesting engine, 7-rule anti-overfitting audit, Kelly criterion position sizing, ATR-based trade management, mean-variance portfolio optimisation, Fama-French factor attribution, and full strategy audit pipeline.



Week 4 — Advanced Topics



Machine learning signals on daily price data, execution algorithms and TWAP/VWAP, crypto momentum strategies, paper trading engine, limit order book simulation, professional VaR/CVaR/stress testing, alternative data (FRED macro pipeline), and final portfolio assembly.



Final Portfolio (2018-2024)



60% SMA+RSI2+VolTgt (SPY) / 25% Equal-Weight Multi-Asset / 15% BTC Momentum



Metric	Final Portfolio	Buy \& Hold SPY

Sharpe Ratio	0.49	0.45

Max Drawdown	-15.41%	-35.75%

Ann. Return	6.08%	11.21%

Ann. Vol	8.34%	20.47%

Applied Forex Research



Post-programme applied research on live trading instruments using real broker data and the Twelve Data API.



LIP-SIM Fusion Strategy



A proprietary indicator combining Lipschutz institutional positioning principles with Simons-inspired systematic filters. Built in Pine Script v5 for TradingView with a complete Python backtesting engine.



Signal logic: EMA 8/21/89 trend confirmation + RSI(14) + MACD(8,17,9) + ADX(14) ≥ 15 + three entry triggers (EMA cross / MACD cross / RSI cross 50) + optional HTF bias filter.



Backtest	Period	Signals	Expectancy	Total R

GLD H1 (EMA cross only)	2yr	38	+1.41R	+24R

XAU/USD H1 (all triggers)	7mo	144	-0.02R	-1R

SMC Mechanical Detector



Complete algorithmic implementation of Smart Money Concepts on 5-minute XAU/USD data:



Swing high/low detection (lookback=5)

Break of Structure (BOS) and Change of Character (CHoCH)

Order Block identification (last opposing candle before BOS)

Fair Value Gap detection (3-bar imbalance, min 0.05%)

Strategy	Expectancy	Total R

SMC alone	-0.07R	-5R

LIP-SIM alone	-0.02R	-1R

SMC + LIP-SIM combined	+0.30R	+7R

Powell 10AM Strategy



PO3-based time strategy anchored at 10:00 AM NY (14:00 UTC DST / 15:00 UTC STD). Tested on XAU/USD 5-minute data with DST-aware session detection.



Best configuration: SHORT only + manipulation > 0.10% + 4-hour hold window



32 signals over 6 months

38% win rate

+0.333R expectancy

+10.6R total

Live Trading Results (16 Real XAU/USD Trades)

Metric	Result

Win rate	62.5%

Avg win	+2.46R

Avg loss	-1.00R (perfect stop discipline)

Expectancy	+1.91R per trade

Total R	+18.58R

Best trade	+4.56R



Every loss is exactly -1.00R — stops are never moved. This discipline combined with an avg win of 2.46R produces the +1.91R expectancy.



The gap between mechanical (+0.30R) and live (+1.91R) is the discretionary SMC overlay — approximately +1.61R per trade of skill premium, now quantified for the first time.



TradingView Indicators (Pine Script v5)



Two paste-ready indicators built for live trading:



1\. Triple Confluence Indicator Combines Powell 10AM + H1 LIP-SIM bias + SMC BOS/CHoCH into a single signal. Shows TC SELL/BUY labels when all three layers align simultaneously. Includes full dashboard, SL/TP lines, session background highlighting, and alerts.



2\. Powell 10AM Indicator Standalone PO3 strategy indicator. Draws the 10AM anchor line, manipulation range box, reversal signal, and SL/TP levels automatically. DST-aware (auto-detects NY Standard vs Daylight Saving time). Full dashboard showing anchor price, manipulation size, direction, and signal status.



Both indicators are UTC-offset configurable and built for XAU/USD on 5-minute charts.



The 27-Tool Utils Library



A reusable Python library covering the complete quantitative trading workflow:



Data and Analysis: analyze\_stock, distribution\_analysis, timeseries, indicators, momentum\_screener



Strategy Building: sma\_backtest, mean\_reversion, pairs\_trading, rsi2\_strategy, volatility\_strategies, multi\_signal



Validation: backtest\_framework, overfitting\_tests, kelly\_sizing, trade\_management, portfolio\_optimizer, factor\_attribution



Advanced: ml\_signals, execution\_algos, crypto\_strategies, paper\_trader, market\_microstructure, risk\_systems, alt\_data, final\_portfolio



Forex: trade\_journal\_analyser, position\_sizer, live\_tracker



Setup

bash

\# Clone the repository

git clone https://github.com/JUSTOH-CPP/quant-trading.git

cd quant-trading



\# Create conda environment

conda create -n quant python=3.11

conda activate quant



\# Install dependencies

pip install numpy pandas matplotlib scipy scikit-learn

pip install yfinance requests jupyter



Data sources used:



yfinance — SPY, GLD, BTC-USD, ETFs (free)

Twelve Data API — XAU/USD H1 and 5-minute (free tier)

FRED API — macro data, yield curve (free, no key required)



TradingView — Pine Script v5 indicators (free account sufficient)



What's Honest and What Isn't



Most trading repositories show cherry-picked results. This one doesn't.



What worked:



Vol targeting beats buy and hold on Sharpe (0.62 vs 0.58)

SMC + LIP-SIM fusion creates +0.30R mechanical edge

Powell 10AM SHORT setup: +0.333R on XAU/USD 5-minute

Live discretionary trading: +1.91R expectancy over 16 trades



What didn't work:



ML on daily price data — regime shift destroyed OOS performance

Pairs trading — only 1 of 45 pairs cointegrated (KO/PEP)

ATR stops on mean reversion — cut returns by 37%

PEAD from price gaps — captured mean reversion, not earnings drift

Optimised SMA parameters — decayed 64% OOS vs unoptimised 50/200

Full 3-layer combined signal (Powell+SMC+LIP-SIM on 5min) — too few signals over 4.5 months to backtest statistically



The honest bottom line: Mechanical signals alone are breakeven on XAU/USD. The live +1.91R edge comes from discretionary SMC judgment on top of systematic filters. The programme quantifies this skill premium for the first time.



Ongoing Work

&#x20;Growing live journal to 50+ trades per pair for statistical significance

&#x20;NAS100 and XAG/USD backtests and strategy research

&#x20;Walk-forward audit on SMC+LIP-SIM system

&#x20;Monte Carlo account growth projector

&#x20;Powell trade journal — 15 trades logged, targeting 25+

Author



Active forex and futures trader with a systematic, empirically grounded approach. Trades XAU/USD, XAG/USD, NAS100, GBP/USD, and GBP/JPY primarily on H1 and M15 timeframes using a fusion of Smart Money Concepts and quantitative methods.



GitHub: JUSTOH-CPP

