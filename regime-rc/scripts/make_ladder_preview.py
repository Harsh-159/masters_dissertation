"""The ladder preview: growth of 10,000 for every method built so far.

    python scripts/make_ladder_preview.py            # both universes
    python scripts/make_ladder_preview.py --universe etf10
    python scripts/make_ladder_preview.py --repopulate   # force stream rebuild

What this is, and what it is not
--------------------------------
A single money-growth chart - one line per method, base to latest rung - drawn
on the **common out-of-sample window** every line covers. It is an
*interim visual*, quarantined exactly like the D-series diagnostics:

* it writes to ``results/diagnostics/ladder/`` (never ``results/figures/``);
* it carries no F-number - the figures of record for this comparison are F4/F5
  at P9, computed over ALL folds with confidence intervals;
* it **never touches the registry** - the streams here are seed-0 re-trainings
  used for a picture, not results;
* everything it writes is regenerable and gitignored.

The extension contract (how future rungs join the chart)
--------------------------------------------------------
The chart is drawn from a stream store::

    results/runs/streams/<universe>/r<rung>_<arm>.parquet
        one column ``net_return``, indexed by date
        (concatenated out-of-sample test segments)

Any phase that wants to appear simply drops its stream file there and reruns
``make ladder`` - no code change. Rung 5 at P8, the full-period runs at P9,
a K-ablation at P10: same store, same filename pattern. The plot normalises
every line to 10,000 at the **latest common start** and truncates to the
**earliest common end**, so today it shows the three-fold window rungs 1-4
cover, and it widens automatically as fuller-period streams land. A colour is
already reserved for rung 5.

Current population
------------------
* rung 0: the four baseline streams already saved by P4 (copied in);
* rungs 1, 2, 3a/3b/3c, 4-hmm, 4-vix: retrained here at seed 0 on folds
  F01-F03 (the P7 protocol), test segments concatenated. ~8 minutes once;
  cached in the store afterwards.
"""

from __future__ import annotations

import argparse
import logging
import shutil
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

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("ladder")

STREAMS_ROOT = PROJECT_ROOT / "results" / "runs" / "streams"
OUTPUT_DIR = PROJECT_ROOT / "results" / "diagnostics" / "ladder"
RUNG0_DIR = PROJECT_ROOT / "results" / "runs" / "rung0"
PREVIEW_FOLDS = ["F01","F02","F03","F04","F05","F06","F07","F08","F09","F10"]   # the folds rungs 1-4 exist on today
STARTING_MONEY = 10_000.0

# Sort rank and the STABLE style per line. Baselines are thin and muted so the
# rungs read as the increments on top of them; rung 5's colour is reserved so
# P8 slots in without restyling anything.
LINE_STYLES = {
    # rung 0 - the base
    ("0", "one_over_n"):   dict(rank=0, colour="#7F7F7F", width=1.0, dash="-",  label="r0 1/N"),
    ("0", "buy_hold_spy"): dict(rank=1, colour="#7F7F7F", width=1.0, dash="--", label="r0 buy&hold SPY"),
    ("0", "inverse_vol"):  dict(rank=2, colour="#B0B0B0", width=1.0, dash="-",  label="r0 inverse-vol"),
    ("0", "mean_variance"):dict(rank=3, colour="#B0B0B0", width=1.0, dash="--", label="r0 Ledoit-Wolf MV"),
    # the increments
    ("1", "single_head"):  dict(rank=4, colour="#0072B2", width=1.9, dash="-",  label="r1 single head"),
    ("2", "moe_k3"):       dict(rank=5, colour="#009E73", width=1.9, dash="-",  label="r2 mixture K=3"),
    ("3a", "sent_gate"):   dict(rank=6, colour="#E69F00", width=1.7, dash="-",  label="r3a signals->gate"),
    ("3b", "sent_experts"):dict(rank=7, colour="#D55E00", width=1.4, dash="--", label="r3b signals->experts"),
    ("3c", "sent_both"):   dict(rank=8, colour="#CC79A7", width=1.4, dash="--", label="r3c signals->both"),
    ("4", "hmm_gate"):     dict(rank=9, colour="#56B4E9", width=1.7, dash="-",  label="r4 HMM gate"),
    ("4", "vix_gate"):     dict(rank=10, colour="#F0E442", width=1.7, dash="-", label="r4 VIX gate"),
    # reserved for the future - P8 drops r5_nsga.parquet in and reruns:
    ("5", "nsga"):         dict(rank=11, colour="#000000", width=2.2, dash="-", label="r5 NSGA-II front"),
}
FALLBACK_STYLE = dict(rank=99, colour="#444444", width=1.4, dash=":", label=None)


# --------------------------------------------------------------------------- #
# Populating the stream store (seed 0, folds F01-F03, NO registry writes)
# --------------------------------------------------------------------------- #
def store_dir(universe: str) -> Path:
    directory = STREAMS_ROOT / universe
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def copy_rung0_streams(universe: str) -> None:
    """Bring P4's saved baseline streams into the store (full test era)."""
    for arm in ("one_over_n", "buy_hold_spy", "inverse_vol", "mean_variance"):
        source = RUNG0_DIR / f"{universe}_{arm}.parquet"
        target = store_dir(universe) / f"r0_{arm}.parquet"
        if source.is_file() and not target.is_file():
            shutil.copyfile(source, target)


def concatenated_test_stream(per_fold_returns: list[pd.Series]) -> pd.DataFrame:
    stream = pd.concat(per_fold_returns).sort_index()
    return stream.to_frame(name="net_return")


def populate_model_streams(universe: str, repopulate: bool) -> None:
    """Retrain seed-0 arms on the preview folds and persist their test streams.

    Deliberately does NOT go through run_rung3.run_arm: that function writes
    registry rows, and a picture must never inflate the trial count.
    """
    wanted = [("1", "single_head")]
    wanted += list(LEARNED_ARMS)                       # r2 + the three r3 arms
    wanted += [("4", arm) for arm in STATISTICAL_ARMS] # the two r4 gates
    target_files = {
        (rung, arm): store_dir(universe) / f"r{rung}_{arm}.parquet"
        for rung, arm in wanted
    }
    missing = {
        key: path for key, path in target_files.items()
        if repopulate or not path.is_file()
    }
    if not missing:
        logger.info("[%s] stream store already populated.", universe)
        return

    logger.info("[%s] populating %d stream(s) (seed 0, folds %s)...",
                universe, len(missing), "/".join(PREVIEW_FOLDS))
    config = load_yaml(PROJECT_ROOT / "configs" / "rung3.yaml")
    artifacts = load_universe_artifacts(
        PROJECT_ROOT / config["paths"]["processed_dir"], universe
    )
    cached_states = find_cached_states(
        PROJECT_ROOT / config["paths"]["states_dir"], universe
    )
    folds = [f for f in folds_for(universe) if f.name in PREVIEW_FOLDS]
    packs_by_fold = {
        fold.name: build_fold_packs(artifacts, cached_states, fold) for fold in folds
    }

    for (rung, arm), path in missing.items():
        per_fold = []
        for fold in folds:
            packs = packs_by_fold[fold.name]

            if (rung, arm) == ("1", "single_head"):
                result = train_single_head(
                    packs["train"], packs["val"],
                    TrainConfig(lambda_cvar=1.0, seed=0),
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

            per_fold.append(evaluate_on_pack(model, packs["test"])["net_returns"])

        concatenated_test_stream(per_fold).to_parquet(path)
        logger.info("  wrote %s", path.name)


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def load_streams(universe: str) -> dict[tuple[str, str], pd.Series]:
    streams: dict[tuple[str, str], pd.Series] = {}
    for path in sorted(store_dir(universe).glob("r*_*.parquet")):
        rung, arm = path.stem.split("_", 1)
        streams[(rung.lstrip("r"), arm)] = pd.read_parquet(path)["net_return"]
    return streams


def plot_ladder(universe: str, streams: dict[tuple[str, str], pd.Series]) -> Path:
    """One growth-of-10,000 line per method, on the common window."""
    common_start = max(series.index.min() for series in streams.values())
    common_end = min(series.index.max() for series in streams.values())

    figure, axis = plt.subplots(figsize=(13, 6.5))
    ordered = sorted(
        streams.items(),
        key=lambda item: LINE_STYLES.get(item[0], FALLBACK_STYLE)["rank"],
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
            path_effects=None,
        )

    axis.axhline(STARTING_MONEY, color="grey", lw=0.8, ls=":")
    axis.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    axis.set_ylabel(f"growth of {STARTING_MONEY:,.0f} (net of 5 bp costs)")
    axis.set_xlabel("date")
    axis.set_title(
        f"Ladder preview - {universe}: every method so far, growth of "
        f"{STARTING_MONEY:,.0f}\n"
        f"common out-of-sample window {common_start.date()} to {common_end.date()} "
        "(seed 0; widens automatically as fuller streams land) - "
        "INTERIM VISUAL, superseded by F4/F5 at P9",
        fontsize=10, loc="left",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
    figure.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"ladder_preview_{universe}.png"
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interim ladder money-growth preview.")
    parser.add_argument("--universe", default="all", help="etf10, sector11, or all.")
    parser.add_argument("--repopulate", action="store_true",
                        help="Rebuild the model streams even if cached.")
    args = parser.parse_args(argv)

    universes = ["etf10", "sector11"] if args.universe == "all" else [args.universe]
    for universe in universes:
        copy_rung0_streams(universe)
        populate_model_streams(universe, repopulate=args.repopulate)
        streams = load_streams(universe)
        path = plot_ladder(universe, streams)
        logger.info("[%s] %d lines -> %s", universe, len(streams), path)

    logger.info("Add a future rung: drop r<rung>_<arm>.parquet into "
                "results/runs/streams/<universe>/ and rerun `make ladder`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
