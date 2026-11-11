"""reads config.yaml into one settings object."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Member:
    """one row of the dow membership schedule. member_to is None if still a member."""

    ticker: str
    member_from: dt.date
    member_to: dt.date | None


@dataclass(frozen=True)
class Universe:
    """a named set of assets. etf and sector use tickers, dow uses a schedule."""

    name: str
    description: str
    tickers: tuple[str, ...] = ()
    schedule: tuple[Member, ...] = ()

    @property
    def download_tickers(self) -> tuple[str, ...]:
        # honeywell appears twice in the dow schedule, dropped in 2008 and added
        # back in 2020, so drop repeats before we go and download anything
        names = self.tickers or tuple(m.ticker for m in self.schedule)
        return tuple(dict.fromkeys(names))


@dataclass(frozen=True)
class Config:
    """every setting from config.yaml, flattened."""

    start: dt.date
    end: dt.date
    raw_dir: Path
    processed_dir: Path
    reports_dir: Path
    plots_dir: Path
    auto_adjust_fallback: bool
    retry_attempts: int
    retry_backoff_seconds: float
    polite_delay_seconds: float
    ffill_limit: int
    min_history_days: int
    # None when the validate block is missing from config.yaml. the checks that
    # need them are skipped rather than guessing a threshold
    outlier_sigma: float | None
    extreme_return_abs: float | None
    stale_run_length: int | None
    risk_free_rate_pct: float
    trading_days_per_year: int
    dpi: int
    palette: str
    universes: dict[str, Universe]

    def universe(self, name: str) -> Universe:
        if name not in self.universes:
            known = ", ".join(sorted(self.universes))
            raise KeyError(f"no universe called {name!r}, we have: {known}")
        return self.universes[name]


def load_config(path: str | Path) -> Config:
    """load config.yaml. paths in it are relative to the project folder."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    root = path.resolve().parent.parent

    run, cache = raw["run"], raw["cache"]
    fetch, clean = raw["fetch"], raw["clean"]
    stats, plots = raw["stats"], raw["plots"]

    # the validate block is optional. comment it out and the soft warnings are
    # skipped. the checks that stop the run do not depend on it
    check = raw.get("validate") or {}

    start, end = _date(run["start"]), _date(run["end"])
    if end <= start:
        raise ValueError(f"run.end ({end}) has to be after run.start ({start})")

    return Config(
        start=start,
        end=end,
        raw_dir=root / cache["raw_dir"],
        processed_dir=root / cache["processed_dir"],
        reports_dir=root / cache["reports_dir"],
        plots_dir=root / cache["plots_dir"],
        auto_adjust_fallback=bool(fetch["auto_adjust_fallback"]),
        retry_attempts=int(fetch["retry_attempts"]),
        retry_backoff_seconds=float(fetch["retry_backoff_seconds"]),
        polite_delay_seconds=float(fetch["polite_delay_seconds"]),
        ffill_limit=int(clean["ffill_limit"]),
        min_history_days=int(clean["min_history_days"]),
        outlier_sigma=_opt(check.get("outlier_sigma"), float),
        extreme_return_abs=_opt(check.get("extreme_return_abs"), float),
        stale_run_length=_opt(check.get("stale_run_length"), int),
        risk_free_rate_pct=_risk_free_rate(stats),
        trading_days_per_year=int(stats["trading_days_per_year"]),
        dpi=int(plots["dpi"]),
        palette=str(plots["palette"]),
        universes=_load_universes(raw["universes"]),
    )


def _load_universes(raw: dict) -> dict[str, Universe]:
    out = {}
    for name, spec in raw.items():
        description = spec.get("description", "")
        if "schedule" in spec:
            members = tuple(
                Member(
                    ticker=row["ticker"],
                    member_from=_date(row["member_from"]),
                    member_to=_date(row["member_to"]) if row.get("member_to") else None,
                )
                for row in spec["schedule"]
            )
            out[name] = Universe(name, description, schedule=members)
        else:
            out[name] = Universe(name, description, tickers=tuple(spec["tickers"]))
    return out


def _opt(value, cast):
    return None if value is None else cast(value)


def _risk_free_rate(stats: dict) -> float:
    """either name works. the number is a percent, so 4.0 means 4%.

    not defaulted on purpose. a missing rate silently becoming zero is what made
    every sharpe in the old reports about a third too high.
    """
    for key in ("risk_free_rate_pct", "risk_free_rate"):
        if key in stats:
            return float(stats[key])
    raise KeyError("stats needs risk_free_rate, written as a percent so 4.0 means 4%")


def _date(value) -> dt.date:
    # yaml gives us a date, a datetime or a string depending on how it was quoted
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))
