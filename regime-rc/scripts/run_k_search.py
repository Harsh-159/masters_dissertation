"""Search the two structural knobs: K (experts) and K_card (names held).

    python scripts/run_k_search.py --universe sector11

Why these two, and why now
--------------------------
**K** was frozen at 3 by gate G2 for arm comparability, never searched. Two
measurements since then argue that was premature on this universe: G2's own
verdict for sector11 was a *negative* finding (experts near-identical, pairwise
correlation 0.923), and the ladder comparison showed the mixture LOSING to the
single head (0.135 versus 0.231 concatenated). If K = 3 is wrong here, every
mixture result on this universe is being read at the wrong capacity.

**K_card** is the whole of rung 5 now that the NSGA front was retired - rung 5
is exactly "keep the trained blend's top-K_card names". The plan lists {3, 5, 7}
as its range; only 5 was ever run.

The design, and why it is shaped this way
------------------------------------------
K_card requires **no retraining**: it re-repairs an already-trained blend. So one
training serves every K_card value, and the grid is essentially free in that
dimension. Compute is spent on K alone.

K is integer and already dense over 1..6, so later rounds cannot usefully refine
*resolution* - there is nothing between 3 and 4. What they can buy is
**replication**: seed-to-seed spread here is 0.02-0.14 while the effects of
interest are a few tenths, so the honest way to separate candidates is more
seeds, not a finer mesh. The three rounds are therefore successive replication:

    round 1  screen    K x K_card, 2 seeds,  8 folds
    round 2  replicate top 3 K values, 5 seeds,  8 folds
    round 3  confirm   top 2 K values, 5 seeds, 12 folds (wider era)

Each round appends its table and its verdict to ``SEARCH_FINDINGS.md`` as it
completes, so the progression is a record rather than something reconstructed
afterwards.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rrc.backtest.metrics import excess_sharpe  # noqa: E402
from rrc.eval.registry import RegistryRow, append_row, config_hash  # noqa: E402
from rrc.eval.slicing import folds_for  # noqa: E402
from rrc.optim.cardinality import cardinality_schedule  # noqa: E402
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts  # noqa: E402
from rrc.optim.train_loop import MixtureTrainConfig, train_mixture  # noqa: E402
from rrc.utils.io import load_yaml  # noqa: E402
from run_rung1 import find_cached_states  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("k-search")

FINDINGS = PROJECT_ROOT / "SEARCH_FINDINGS.md"
UNCONSTRAINED = 0          # K_card sentinel: hold every active name


def evaluate_k(
    universe: str, k: int, k_cards: list[int], config: dict,
    artifacts, states, folds, seeds: list[int],
) -> dict[int, tuple[float, float]]:
    """Train at one K; score every K_card from the SAME trained models.

    Returns ``{k_card: (mean concatenated Sharpe, seed sd)}``. The inner loop is
    free: ``cardinality_schedule`` only re-repairs weights that already exist.
    """
    loss = config["loss"]
    packs_by_fold = {f.name: build_fold_packs(artifacts, states, f) for f in folds}
    streams: dict[int, list[list[np.ndarray]]] = {kc: [] for kc in k_cards}
    risk_frees: list[list[np.ndarray]] = []

    for seed in seeds:
        per_kcard = {kc: [] for kc in k_cards}
        per_seed_rf = []
        for fold in folds:
            packs = packs_by_fold[fold.name]
            result = train_mixture(
                packs["train"], packs["val"],
                MixtureTrainConfig(
                    n_experts=k,
                    lambda_cvar=float(loss["lambda_cvar"]),
                    lambda_balance=float(loss["lambda_balance"]),
                    lambda_turn=float(loss["lambda_turn"]),
                    lambda_div=float(loss["lambda_div"]),
                    seed=seed,
                ),
            )
            val = packs["val"]
            with torch.no_grad():
                blend = result.model.weights_for_pack(val).numpy()
            for kc in k_cards:
                schedule = blend if kc == UNCONSTRAINED else cardinality_schedule(
                    blend, val, artifacts, kc
                )
                per_kcard[kc].append(_daily_stream(schedule, val))
            per_seed_rf.append(val.risk_free.numpy())
        for kc in k_cards:
            streams[kc].append(per_kcard[kc])
        risk_frees.append(per_seed_rf)

    scores = {}
    for kc in k_cards:
        per_seed = [
            excess_sharpe(np.concatenate(seed_streams), np.concatenate(seed_rf))
            for seed_streams, seed_rf in zip(streams[kc], risk_frees)
        ]
        spread = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
        scores[kc] = (float(np.mean(per_seed)), spread)
    return scores


def _daily_stream(schedule, pack) -> np.ndarray:
    """Daily net returns of a weight schedule through the shared ledger."""
    from rrc.backtest.ledger import run_ledger

    result = run_ledger(
        targets=torch.as_tensor(np.asarray(schedule), dtype=torch.float64),
        exec_days=pack.execution_positions,
        simple_returns=pack.returns,
        cash_index=pack.cash_index,
    )
    return result.net_returns.detach().numpy()


def run_round(
    label: str, universe: str, k_values: list[int], k_cards: list[int],
    seeds: list[int], fold_names: list[str], config: dict,
    artifacts, states, registry_path: Path, dry_run: bool,
) -> pd.DataFrame:
    """Score a (K x K_card) block and return tidy rows."""
    folds = [f for f in folds_for(universe) if f.name in fold_names]
    logger.info("--- %s: K=%s x K_card=%s | %d folds, %d seeds ---",
                label, k_values, k_cards, len(folds), len(seeds))

    rows = []
    for k in k_values:
        scores = evaluate_k(universe, k, k_cards, config, artifacts, states, folds, seeds)
        for k_card, (sharpe, spread) in scores.items():
            rows.append({"round": label, "K": k, "K_card": k_card,
                         "sharpe": sharpe, "sharpe_sd": spread,
                         "n_folds": len(folds), "n_seeds": len(seeds)})
            if not dry_run:
                identity = {"universe": universe, "rung": "k-search", "round": label,
                            "K": k, "K_card": k_card,
                            "loss": {x: config["loss"][x] for x in sorted(config["loss"])}}
                append_row(
                    RegistryRow(
                        universe=universe, rung="k-search", arm=f"K{k}_card{k_card}",
                        seed=seeds[0], fold=label, split="val",
                        n_params=0, cfg_hash=config_hash(identity),
                        excess_sharpe=sharpe,
                        note=(f"{label}; K={k}; K_card="
                              f"{'unconstrained' if k_card == UNCONSTRAINED else k_card}; "
                              f"{len(folds)} folds x {len(seeds)} seeds"),
                    ),
                    registry_path,
                )
        best_kc = max(scores, key=lambda c: scores[c][0])
        logger.info("  K=%d: best K_card=%s -> %+.3f (sd %.3f)", k,
                    "none" if best_kc == UNCONSTRAINED else best_kc,
                    scores[best_kc][0], scores[best_kc][1])
    return pd.DataFrame(rows)


def append_findings(universe: str, label: str, table: pd.DataFrame, verdict: str) -> None:
    """Append one round's table and verdict to the findings log."""
    if not FINDINGS.exists():
        FINDINGS.write_text(
            "# Search findings\n\n"
            "A running record of the hyperparameter searches, written by the search\n"
            "scripts as each round completes so the progression is a record rather\n"
            "than something reconstructed afterwards. Every configuration listed here\n"
            "is a deflated-Sharpe trial.\n\n"
            "Metric throughout: **concatenated validation excess Sharpe** - each\n"
            "seed's fold validation segments stitched into one continuous daily\n"
            "stream and scored once, the same way the baselines and the literature\n"
            "are scored. Averaging per-fold Sharpes is not used: it rebases wealth\n"
            "each January and differed from this by 0.48 on sector11.\n"
        )

    pivot = table.pivot(index="K", columns="K_card", values="sharpe").round(3)
    pivot.columns = ["unconstrained" if c == UNCONSTRAINED else f"K_card={c}"
                     for c in pivot.columns]
    spreads = table.pivot(index="K", columns="K_card", values="sharpe_sd").round(3)

    lines = [
        f"\n---\n\n## {universe} - {label}",
        f"\n_{dt.datetime.now():%Y-%m-%d %H:%M} | "
        f"{table['n_folds'].iloc[0]} folds x {table['n_seeds'].iloc[0]} seeds | "
        f"{len(table)} configurations_\n",
        "\n### Concatenated validation Sharpe\n",
        pivot.to_markdown(),
        f"\n\nSeed spread (sd) ranged {spreads.min().min():.3f}-{spreads.max().max():.3f}; "
        "treat differences smaller than this as unresolved.\n",
        f"\n**Verdict.** {verdict}\n",
    ]
    with FINDINGS.open("a") as handle:
        handle.write("\n".join(lines))
    logger.info("  findings -> %s", FINDINGS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search K and K_card.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "k_search.yaml"))
    parser.add_argument("--universe", required=True, choices=["etf10", "sector11"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_yaml(Path(args.config))
    rounds = config["rounds"]
    k_cards = [int(c) for c in config["k_cards"]]
    registry_path = PROJECT_ROOT / "results" / "registry.csv"

    artifacts = load_universe_artifacts(
        PROJECT_ROOT / config["paths"]["processed_dir"], args.universe
    )
    states = find_cached_states(
        PROJECT_ROOT / config["paths"]["states_dir"], args.universe
    )

    all_rows = []

    # --- round 1: screen the full K range -------------------------------
    spec = rounds["screen"]
    table = run_round("round 1 (screen)", args.universe,
                      [int(k) for k in spec["k_values"]], k_cards,
                      [int(s) for s in spec["seeds"]], list(spec["fold_names"]),
                      config, artifacts, states, registry_path, args.dry_run)
    all_rows.append(table)
    best = table.loc[table["sharpe"].idxmax()]
    top_k = (table.groupby("K")["sharpe"].max()
             .sort_values(ascending=False).index.tolist())
    append_findings(
        args.universe, "round 1 (screen)", table,
        f"Best cell K={int(best.K)}, K_card="
        f"{'unconstrained' if best.K_card == UNCONSTRAINED else int(best.K_card)}"
        f" at {best.sharpe:+.3f}. Carrying K in {top_k[:3]} to round 2 for replication; "
        "at two seeds these differences are screening signal only.",
    )

    # --- round 2: replicate the leaders ---------------------------------
    spec = rounds["replicate"]
    table = run_round("round 2 (replicate)", args.universe, top_k[:3], k_cards,
                      [int(s) for s in spec["seeds"]], list(spec["fold_names"]),
                      config, artifacts, states, registry_path, args.dry_run)
    all_rows.append(table)
    best = table.loc[table["sharpe"].idxmax()]
    top_k2 = (table.groupby("K")["sharpe"].max()
              .sort_values(ascending=False).index.tolist())
    append_findings(
        args.universe, "round 2 (replicate)", table,
        f"With {spec['seeds']} seeds the leader is K={int(best.K)}, K_card="
        f"{'unconstrained' if best.K_card == UNCONSTRAINED else int(best.K_card)}"
        f" at {best.sharpe:+.3f}. More seeds buy resolution here; a finer K mesh "
        "cannot, since K is integer and already dense.",
    )

    # --- round 3: confirm on a wider era ---------------------------------
    spec = rounds["confirm"]
    table = run_round("round 3 (confirm)", args.universe, top_k2[:2], k_cards,
                      [int(s) for s in spec["seeds"]], list(spec["fold_names"]),
                      config, artifacts, states, registry_path, args.dry_run)
    all_rows.append(table)
    best = table.loc[table["sharpe"].idxmax()]
    append_findings(
        args.universe, "round 3 (confirm)", table,
        f"On the wider {spec['fold_names'][0]}-{spec['fold_names'][-1]} era the "
        f"choice is K={int(best.K)}, K_card="
        f"{'unconstrained' if best.K_card == UNCONSTRAINED else int(best.K_card)}"
        f" at {best.sharpe:+.3f} (sd {best.sharpe_sd:.3f}). "
        "This is the value to freeze, subject to the usual caveat that it was "
        "selected on validation and still owes a held-out check.",
    )

    combined = pd.concat(all_rows, ignore_index=True)
    runs_dir = PROJECT_ROOT / config["paths"]["runs_dir"]
    runs_dir.mkdir(parents=True, exist_ok=True)
    combined.round(4).to_csv(runs_dir / f"k_search_{args.universe}.csv", index=False)

    logger.info("=" * 72)
    logger.info("FINAL: K=%d, K_card=%s -> %+.3f (sd %.3f)",
                int(best.K),
                "unconstrained" if best.K_card == UNCONSTRAINED else int(best.K_card),
                best.sharpe, best.sharpe_sd)
    logger.info("  %d configurations scored across 3 rounds", len(combined))
    logger.info("  findings log: %s", FINDINGS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
