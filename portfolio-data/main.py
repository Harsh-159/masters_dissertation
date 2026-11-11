"""builds the price and return panels for each universe.

edit the settings below and run the file:

    ./.venv/bin/python main.py

everything else lives in config/config.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import load_config
from src.fetcher import DataFetcher
from src.processor import DataProcessor
from src.validator import validate
from src.visualizer import Visualizer

# settings

# which universes to build. drop any of them from the list to skip it
universes = ["etf", "sector", "dow"]

# ignore what is already downloaded and pull it all from yahoo again. slow, and
# only needed if you think the cache is stale or a dead ticker came back
force_refresh = False

# set to False to write only the parquet files and skip the two figures
make_plots = True

# ---------------------------------------------------------------------------

config_path = Path(__file__).resolve().parent / "config" / "config.yaml"
log = logging.getLogger("portfolio-data")


def run_one(cfg, name: str, fetcher: DataFetcher, visualizer: Visualizer | None) -> None:
    log.info("=== %s ===", name)
    universe = cfg.universe(name)

    prices = fetcher.fetch_universe(universe.download_tickers, force_refresh)
    panels = DataProcessor(cfg).process(name, prices, universe.download_tickers)
    validate(cfg, panels, universe)

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    panels.prices.to_parquet(cfg.processed_dir / f"{name}_prices.parquet")
    panels.returns.to_parquet(cfg.processed_dir / f"{name}_returns.parquet")

    summary = Visualizer(cfg).summary_stats(panels)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(cfg.reports_dir / f"summary_stats_{name}.csv")

    if visualizer is not None:
        visualizer.render(panels, summary)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    cfg = load_config(config_path)
    log.info("window %s to %s", cfg.start, cfg.end)

    fetcher = DataFetcher(cfg)
    visualizer = Visualizer(cfg) if make_plots else None

    failed = []
    for name in universes:
        try:
            run_one(cfg, name, fetcher, visualizer)
        except Exception as exc:
            # one broken universe should not stop the other two
            failed.append(name)
            log.exception("%s failed: %s", name, exc)

    if failed:
        log.error("done, but these failed: %s", ", ".join(failed))
        return 1
    log.info("done. panels in %s, tables in %s", cfg.processed_dir, cfg.reports_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
