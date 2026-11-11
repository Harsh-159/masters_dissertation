"""The combined ladder: 10,000 split 50/50 across both universes' strategies.

    python scripts/make_combined_ladder.py

What this shows
---------------
One growth-of-10,000 line per method, where the investor puts half the money
into the method's etf10 strategy and half into its sector11 strategy, with the
50/50 split **rebalanced daily** (the blended daily return is the average of
the two strategies' daily returns). This is a *portfolio of strategies* - an
honest presentation aggregate - NOT a joint optimisation over a merged
21-asset universe (which would double-count the S&P: the sector SPDRs are
slices of it, and etf10 holds SPY).

The window-matching detail that makes the blend honest
------------------------------------------------------
Returns can only be blended on the SAME dates. etf10's preview streams cover
its folds F01-F03 (test years 2015-2017); sector11's preview folds test
2007-2009 - no overlap. So this script trains the sector11 arms on the folds
whose test years match etf10's: **F09, F10, F11 (test 2015, 2016, 2017)** -
same walk-forward protocol, seed 0, no registry writes - and blends on the
shared 2015-2017 calendar. Those matched streams live in their own store
(``streams/sector11_matched2015/``) so they never collide with the per-universe
preview streams or any future P9 output.

Quarantine: same rules as the ladder preview - diagnostics directory, no
F-number, no registry, everything regenerable. Superseded by P9's full-period
figures.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rrc.eval.slicing import folds_for  # noqa: E402
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts  # noqa: E402
from rrc.optim.train_loop import (  # noqa: E402
    MixtureTrainConfig,
    TrainConfig,
    evaluate_on_pack,
    train_mixture,
    train_single_head,
)
from rrc.utils.io import load_yaml  # noqa: E402
from run_rung1 import find_cached_states  # noqa: E402
from run_rung3 import (  # noqa: E402
    LEARNED_ARMS,
    STATISTICAL_ARMS,
    attach_fixed_memberships,
    model_factory,
    statistical_memberships,
)
from make_ladder_preview import (  # noqa: E402
    LINE_STYLES,
    FALLBACK_STYLE,
    STARTING_MONEY,
    STREAMS_ROOT,
    OUTPUT_DIR,
    concatenated_test_stream,
    copy_rung0_streams,
    load_streams,
    populate_model_streams,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("combined-ladder")

# sector11 folds whose TEST years are 2015/2016/2017 - the years etf10's
# preview folds test. This is what makes a daily 50/50 blend well-defined.
MATCHED_SECTOR_FOLDS = ["F09", "F10", "F11"]
MATCHED_STORE = STREAMS_ROOT / "sector11_matched2015"


def populate_matched_sector_streams(repopulate: bool) -> None:
    """Train sector11 arms on the 2015-2017-testing folds; persist streams.

    Same protocol as the ladder preview (seed 0, rung-3 config, no registry
    rows), just a different fold triple, stored separately.
    """
    MATCHED_STORE.mkdir(parents=True, exist_ok=True)

    wanted = [("1", "single_head")]
    wanted += list(LEARNED_ARMS)
    wanted += [("4", arm) for arm in STATISTICAL_ARMS]
    def is_complete(rung: str, arm: str) -> bool:
        base = MATCHED_STORE / f"r{rung}_{arm}"
        return base.with_suffix(".parquet").is_file() and base.with_suffix(".meta.json").is_file()

    missing = {
        (rung, arm): MATCHED_STORE / f"r{rung}_{arm}.parquet"
        for rung, arm in wanted
        if repopulate or not is_complete(rung, arm)
    }
    if not missing:
        logger.info("matched sector11 streams already populated.")
        return

    logger.info("populating %d matched sector11 stream(s) (folds %s, seed 0)...",
                len(missing), "/".join(MATCHED_SECTOR_FOLDS))
    config = load_yaml(PROJECT_ROOT / "configs" / "rung3.yaml")
    artifacts = load_universe_artifacts(
        PROJECT_ROOT / config["paths"]["processed_dir"], "sector11"
    )
    cached_states = find_cached_states(
        PROJECT_ROOT / config["paths"]["states_dir"], "sector11"
    )
    folds = [f for f in folds_for("sector11") if f.name in MATCHED_SECTOR_FOLDS]
    packs_by_fold = {
        fold.name: build_fold_packs(artifacts, cached_states, fold) for fold in folds
    }

    for (rung, arm), path in missing.items():
        per_fold = []
        per_fold_turnover = []
        for fold in folds:
            packs = packs_by_fold[fold.name]
            if (rung, arm) == ("1", "single_head"):
                result = train_single_head(
                    packs["train"], packs["val"], TrainConfig(lambda_cvar=1.0, seed=0)
                )
                model = result.head
            else:
                hmm_states = None
                if arm in STATISTICAL_ARMS:
                    per_split, hmm_states, _ = statistical_memberships(
                        arm, artifacts, fold, packs, config
                    )
                    packs = attach_fixed_memberships(packs, per_split)
                factory = model_factory(rung, arm, packs["train"], config, hmm_states)
                result = train_mixture(
                    packs["train"], packs["val"],
                    MixtureTrainConfig(
                        n_experts=int(config["mixture"]["n_experts"]),
                        lambda_cvar=float(config["loss"]["lambda_cvar"]),
                        lambda_balance=float(config["loss"]["lambda_balance"]),
                        seed=0,
                    ),
                    model=factory,
                )
                model = result.model
            test_evaluation = evaluate_on_pack(model, packs["test"])
            per_fold.append(test_evaluation["net_returns"])
            per_fold_turnover.append(test_evaluation["metrics"]["ann_turnover"])

        concatenated_test_stream(per_fold).to_parquet(path)
        # Sidecar: the stream alone cannot yield turnover (that needs the
        # weight schedules), so it is recorded here at population time.
        path.with_suffix("").with_suffix(".meta.json").write_text(
            json.dumps({"ann_turnover_mean": sum(per_fold_turnover) / len(per_fold_turnover)})
        )
        logger.info("  wrote %s (+ meta)", path.name)

    # Rung-0 baselines for the matched window: the P4 streams already cover
    # 2007-2024, so the 2015-2017 slice is taken at blend time - nothing to
    # retrain. Copy them in unchanged for a uniform store layout.
    for arm in ("one_over_n", "buy_hold_spy", "inverse_vol", "mean_variance"):
        source = STREAMS_ROOT / "sector11" / f"r0_{arm}.parquet"
        target = MATCHED_STORE / f"r0_{arm}.parquet"
        if source.is_file() and not target.is_file():
            target.write_bytes(source.read_bytes())


def load_matched_streams() -> dict[tuple[str, str], pd.Series]:
    streams: dict[tuple[str, str], pd.Series] = {}
    for path in sorted(MATCHED_STORE.glob("r*_*.parquet")):
        rung, arm = path.stem.split("_", 1)
        streams[(rung.lstrip("r"), arm)] = pd.read_parquet(path)["net_return"]
    return streams


def blend_streams(
    etf_streams: dict, sector_streams: dict
) -> dict[tuple[str, str], pd.Series]:
    """Daily-rebalanced 50/50 blend per method, on the shared calendar.

    The blended daily return is the plain average of the two strategies'
    returns - which is exactly a portfolio holding half its value in each
    strategy and rebalancing the split back to 50/50 every day.
    """
    blended: dict[tuple[str, str], pd.Series] = {}
    for key in sorted(set(etf_streams) & set(sector_streams)):
        joined = pd.concat(
            [etf_streams[key].rename("etf"), sector_streams[key].rename("sector")],
            axis=1, join="inner",
        ).dropna()
        blended[key] = 0.5 * joined["etf"] + 0.5 * joined["sector"]
    return blended


def plot_combined(blended: dict[tuple[str, str], pd.Series]) -> Path:
    common_start = max(series.index.min() for series in blended.values())
    common_end = min(series.index.max() for series in blended.values())

    figure, axis = plt.subplots(figsize=(13, 6.5))
    ordered = sorted(
        blended.items(), key=lambda item: LINE_STYLES.get(item[0], FALLBACK_STYLE)["rank"]
    )
    for key, series in ordered:
        style = LINE_STYLES.get(key, FALLBACK_STYLE)
        window = series.loc[common_start:common_end]
        money = STARTING_MONEY * (1.0 + window).cumprod()
        axis.plot(
            money.index, money,
            color=style["colour"], lw=style["width"], ls=style["dash"],
            label=style["label"] or f"r{key[0]} {key[1]}",
            zorder=3 if style["width"] > 1.2 else 2,
        )

    axis.axhline(STARTING_MONEY, color="grey", lw=0.8, ls=":")
    axis.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    axis.set_ylabel(f"growth of {STARTING_MONEY:,.0f} (net of 5 bp costs)")
    axis.set_xlabel("date")
    axis.set_title(
        f"Combined ladder - 10,000 split 50/50 across the etf10 and sector11 "
        f"strategies (daily-rebalanced blend)\n"
        f"common out-of-sample window {common_start.date()} to {common_end.date()}; "
        "sector11 side uses folds F09-F11 (test 2015-2017) to match etf10's "
        "F01-F03; seed 0 - INTERIM VISUAL (portfolio of strategies, not a "
        "merged-universe optimisation)",
        fontsize=9.5, loc="left",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
    figure.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "ladder_combined_50_50.png"
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Combined 50/50 two-universe ladder.")
    parser.add_argument("--repopulate", action="store_true")
    args = parser.parse_args(argv)

    # Ensure both ingredient stores exist.
    copy_rung0_streams("etf10")
    copy_rung0_streams("sector11")
    populate_model_streams("etf10", repopulate=False)
    populate_matched_sector_streams(repopulate=args.repopulate)

    etf_streams = load_streams("etf10")
    sector_streams = load_matched_streams()
    blended = blend_streams(etf_streams, sector_streams)

    path = plot_combined(blended)
    logger.info("%d blended lines -> %s", len(blended), path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
