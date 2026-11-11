# portfolio-data

downloads daily prices for three asset universes, cleans them, checks them, and
writes a price panel and a log return panel for each.

## running it

open `main.py`, edit the three settings at the top, then:

```bash
./.venv/bin/python main.py
```

everything else is in `config/config.yaml`.

## what it writes

| file | what it is |
|---|---|
| `data/processed/<u>_prices.parquet` | daily adjusted prices, dates by tickers |
| `data/processed/<u>_returns.parquet` | daily log returns, same shape |
| `reports/summary_stats_<u>.csv` | annual return, volatility, sharpe, skew, kurtosis, max drawdown |
| `plots/<u>/cumulative_growth.png` | growth of 1.0, linear and log |
| `plots/<u>/summary_table.png` | the stats table as an image |

raw downloads are cached in `data/raw/<TICKER>.parquet`. the cache holds
everything yahoo has, so changing the dates in the config never re-downloads.

## the three universes

- **etf** ten global etfs, diversified and mostly uncorrelated
- **sector** eleven spdr sector etfs, two of which start late (xlre 2015, xlc 2018)
- **dow** every djia member at any point in 2007 to 2025, with the dates each
  one joined and left

## the rules that matter

- prices are adjusted for splits and dividends
- assets line up on the days something actually traded, not a generated
  business day calendar
- gaps are filled forwards only, up to five days. filling backwards would copy a
  2015 price into 2007
- an asset that did not exist yet stays empty. nothing is invented
- returns are `ln(P_t / P_t-1)`, and the first day of each asset is empty rather
  than zero

six checks stop the run if the data is wrong: duplicate or out of order dates,
infinities, prices at or below zero, prices before an asset existed, short gaps
that should have been filled, and returns without a price on both sides.

extreme returns, frozen prices and short histories only produce a warning. a 20%
day in october 2008 is real, and throwing it away would be worse than keeping it.

## the dow has a hole in it, on purpose

four symbols cannot be retrieved at all: `UTX` became `RTX` in 2020, `KFT` split
up in 2012, `DWDP` only existed for two years, and `WBA` was taken private in
2025. old `GM` went bankrupt in 2009 and yahoo only has the company that
relisted in 2010, so none of its dow years are covered.

the run warns about each of these by name. this is survivorship bias, and free
data cannot fix it. it can only admit it.
