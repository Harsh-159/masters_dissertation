# portfolio-forecast

the baseline. one xgboost model per asset tries to predict that asset's next day
return, and a simple rule trades on the prediction. this is the floor the
reservoir and lstm models have to clear.

## running it

open `baseline_xgboost.py`, edit the settings at the top, then run it.

```bash
./.venv/bin/python baseline_xgboost.py
```

it reads `../portfolio-data/data/processed/<universe>_returns.parquet`, so run
that project first.

## the method

- one model per asset, predicting tomorrow's log return
- 18 features, all things you could know at today's close: the last ten daily
  returns, rolling average and volatility over 5, 10 and 21 days, and total
  return over 21 and 63 days
- first 75% of each asset's days train, last 25% test, in date order. never
  shuffled, because shuffling puts 2024 in the training set and 2019 in the test
  set and invents skill the model does not have
- the trade is long if it predicts up, short if it predicts down. shorting keeps
  this a test of the forecast rather than a test of being in the market
- no hyperparameter tuning. tuning a baseline until it looks good defeats the
  point of having one

## results

test window april 2021 to december 2025, risk free rate 4%.

| universe | assets | dir acc | IC | sharpe | vs b&h | growth | vs b&h |
|---|---|---|---|---|---|---|---|
| etf | 10 | 50.6% | 0.010 | -0.18 | 0.07 | 1.15x | 1.25x |
| sector | 11 | 51.0% | 0.011 | -0.05 | 0.39 | 1.18x | 1.58x |
| dow | 45 | 50.0% | 0.017 | -0.28 | 0.37 | 1.12x | 1.58x |

directional accuracy near 50% is a coin flip and an IC near 0 means predictions
and reality barely relate. **this is the expected, honest result.** daily returns
are close to unpredictable, and a baseline showing real skill would more likely
mean a look-ahead bug than a discovery.

the strategy loses to buy and hold on every universe, on both risk adjusted
return and money. four dow tickers are skipped automatically because they have no
data at all: UTX, KFT, WBA and DWDP.

## what it writes

| file | what it is |
|---|---|
| `reports/metrics_<u>.csv` | one row per asset |
| `reports/summary_<u>.json` | the whole universe rolled up, plus the top features |
| `plots/<u>_equity.png` | growth of 1.0 out of sample, strategy against buy and hold |
