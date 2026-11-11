"""xgboost baseline. can a standard model predict tomorrow's return?

one model per asset, trained on the last few weeks of that asset's own returns,
predicting the next day. then a simple rule: go long if it says up, short if it
says down, and see whether that beats just holding.

this is the floor the reservoir and lstm models have to clear. it is meant to be
unimpressive.

edit the settings below and run the file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

# which universes to run. drop any of them from the list to skip it
universes = ["etf", "sector", "dow"]

# where portfolio-data left the return panels. run that first if they are missing
returns_dir = Path("../portfolio-data/data/processed")

# how far back the features reach. lag 1 is today's return, which we do know at
# today's close, lag 2 is yesterday's, and so on
return_lags = 10

# rolling average and volatility over a week, two weeks and a month
roll_windows = [5, 10, 21]

# total return over the last month and the last quarter
momentum_windows = [21, 63]

# first 75% of each asset's days to train, last 25% to test, in date order.
# shuffling would put 2024 in the training set and 2019 in the test set, which
# hands the model the future and invents skill it does not have
train_fraction = 0.75

# skip an asset that cannot fill both sides of the split
min_train_days = 250
min_test_days = 60

# fixed xgboost settings, deliberately not tuned. tuning a baseline until it
# looks good defeats the point of having one
model_params = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "random_state": 42,
}

# annual risk free rate as a percent, so 4.0 means 4%
risk_free_rate_pct = 4.0
trading_days_per_year = 252

reports_dir = Path("reports")
plots_dir = Path("plots")

# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("baseline")

here = Path(__file__).resolve().parent


def build_features(returns: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """turn one asset's returns into a table of features and next day answers.

    every column is something you could have computed at today's close. the
    target is tomorrow, which you could not.
    """
    table = pd.DataFrame(index=returns.index)
    for lag in range(1, return_lags + 1):
        table[f"ret_lag{lag}"] = returns.shift(lag - 1)
    for window in roll_windows:
        table[f"roll_mean{window}"] = returns.rolling(window).mean()
        table[f"roll_std{window}"] = returns.rolling(window).std()
    for window in momentum_windows:
        table[f"mom{window}"] = returns.rolling(window).sum()

    table["target"] = returns.shift(-1)
    # drops the warm up rows at the start, any leading gap before the asset
    # existed, and the last row where tomorrow is unknown
    table = table.dropna()
    return table.drop(columns="target"), table["target"]


def annual_sharpe(daily: pd.Series) -> float:
    daily = daily.dropna()
    spread = daily.std(ddof=1) if len(daily) > 1 else 0.0
    if daily.empty or spread == 0:
        return float("nan")
    daily_rf = (risk_free_rate_pct / 100.0) / trading_days_per_year
    return float((daily.mean() - daily_rf) / spread * np.sqrt(trading_days_per_year))


def growth(daily: pd.Series) -> float:
    """what 1.0 turns into. these are log returns so they add up."""
    return float(np.exp(daily.dropna().sum()))


def score_asset(ticker: str, actual: pd.Series, predicted: np.ndarray) -> tuple[dict, pd.Series]:
    """how good the forecast was, and what trading on it would have earned."""
    predicted = pd.Series(predicted, index=actual.index)
    error = predicted - actual

    # long when it predicts up, short when it predicts down. shorting is what
    # makes this a forecast test rather than a market exposure test
    strategy = np.sign(predicted) * actual

    row = {
        "ticker": ticker,
        "n_test": len(actual),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        # how often it got the direction right. 0.5 is a coin flip
        "dir_acc": float((np.sign(predicted) == np.sign(actual)).mean()),
        # how well predictions line up with what happened. 0 is no signal
        "ic": float(predicted.corr(actual)) if predicted.std() > 0 else float("nan"),
        "sharpe_strategy": annual_sharpe(strategy),
        "sharpe_buyhold": annual_sharpe(actual),
    }
    return row, strategy


def run_universe(name: str) -> dict | None:
    path = (here / returns_dir / f"{name}_returns.parquet").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no returns panel at {path}. run portfolio-data first")

    returns = pd.read_parquet(path)
    log.info("%s: %d assets, %d days", name, returns.shape[1], returns.shape[0])

    rows, strategies, holds, importances = [], {}, {}, {}

    for ticker in returns.columns:
        features, target = build_features(returns[ticker])
        cut = int(len(features) * train_fraction)
        train_x, test_x = features.iloc[:cut], features.iloc[cut:]
        train_y, test_y = target.iloc[:cut], target.iloc[cut:]

        if len(train_x) < min_train_days or len(test_x) < min_test_days:
            log.warning("  %-5s skipped, only %d train and %d test days",
                        ticker, len(train_x), len(test_x))
            continue

        model = XGBRegressor(objective="reg:squarederror", n_jobs=-1, **model_params)
        model.fit(train_x, train_y)

        row, strategy = score_asset(ticker, test_y, model.predict(test_x))
        rows.append(row)
        strategies[ticker] = strategy
        holds[ticker] = test_y
        for feature, gain in zip(features.columns, model.feature_importances_):
            importances.setdefault(feature, []).append(float(gain))

    if not rows:
        log.warning("%s: nothing had enough history", name)
        return None

    metrics = pd.DataFrame(rows).set_index("ticker")
    reports = here / reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(reports / f"metrics_{name}.csv")

    # hold every asset's strategy in equal weight, one line per day
    strategy_daily = pd.DataFrame(strategies).mean(axis=1)
    hold_daily = pd.DataFrame(holds).mean(axis=1)

    summary = {
        "universe": name,
        "n_assets": len(metrics),
        "test_start": str(strategy_daily.index.min().date()),
        "test_end": str(strategy_daily.index.max().date()),
        "mean_mae": float(metrics["mae"].mean()),
        "mean_dir_acc": float(metrics["dir_acc"].mean()),
        "mean_ic": float(metrics["ic"].mean()),
        "sharpe_strategy": annual_sharpe(strategy_daily),
        "sharpe_buyhold": annual_sharpe(hold_daily),
        "growth_strategy": growth(strategy_daily),
        "growth_buyhold": growth(hold_daily),
        "top_features": sorted(
            ((f, float(np.mean(g))) for f, g in importances.items()),
            key=lambda pair: pair[1], reverse=True,
        )[:5],
    }
    (reports / f"summary_{name}.json").write_text(json.dumps(summary, indent=2))
    plot_equity(name, strategy_daily, hold_daily)
    return summary


def plot_equity(name: str, strategy: pd.Series, hold: pd.Series) -> None:
    out_dir = here / plots_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(strategy.index, np.exp(strategy.cumsum()), lw=1.4, label="xgboost sign strategy")
    ax.plot(hold.index, np.exp(hold.cumsum()), lw=1.4, label="equal weight buy and hold")
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_title(f"growth of 1.0 out of sample, {name}")
    ax.set_ylabel("growth of 1.0")
    ax.set_xlabel("date")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_equity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_results(summaries: list[dict]) -> None:
    """the table that actually answers the question."""
    print("\n" + "=" * 78)
    print("BASELINE RESULTS, test set only")
    print("=" * 78)
    print(f"{'universe':<9}{'assets':>7}{'dir acc':>9}{'IC':>8}"
          f"{'sharpe':>9}{'vs b&h':>9}{'growth':>9}{'vs b&h':>9}")
    print("-" * 78)
    for s in summaries:
        print(f"{s['universe']:<9}{s['n_assets']:>7}{s['mean_dir_acc']:>8.1%}"
              f"{s['mean_ic']:>8.3f}{s['sharpe_strategy']:>9.2f}{s['sharpe_buyhold']:>9.2f}"
              f"{s['growth_strategy']:>8.2f}x{s['growth_buyhold']:>8.2f}x")
    print("-" * 78)
    print(f"test window {summaries[0]['test_start']} to {summaries[0]['test_end']}")
    print("\ndirectional accuracy near 50% and IC near 0 is the honest answer here.")
    print("daily returns are close to unpredictable. a baseline showing real skill")
    print("would more likely mean a look-ahead bug than a discovery.")
    print("=" * 78 + "\n")


def main() -> int:
    summaries = []
    for name in universes:
        summary = run_universe(name)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        log.error("nothing ran")
        return 1

    print_results(summaries)
    log.info("per asset numbers in %s, summaries alongside them", reports_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
