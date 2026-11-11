"""Validation-hot is not test-hot: the N_r x rho surface at default nuisances.

    python scripts/make_grid_fine_heatmap.py --states-dir <scratch dir from run_fine_grid>

The fine grid (run_fine_grid.py) scored 810 sector11 cells on validation but only
re-scored the top 12 on test. To draw a clean N_r x rho picture we need test on
the SAME cells as val, so this script fixes the three nuisance axes at the project
defaults (alpha 0.30, std 0.70, tau_g 1.00) - isolating the two axes with real
effect - and computes BOTH numbers on that 6 x 5 slice:

* val  - read from the fine-grid CSV (VAL split, selection folds, 2 seeds).
* test - computed here (TEST split, all folds, 3 seeds), the honest number.

Two heatmaps, shared axes. If the bright cell of each panel is in a different
place, selecting the reservoir on validation points away from the best test
reservoir - the transfer-failure finding (F4/L2) in one picture.

Writes results/runs/grid_fine/. No registry rows, no holdout.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rrc.eval.slicing import folds_for  # noqa: E402
from rrc.optim.fold_data import build_fold_packs, load_universe_artifacts  # noqa: E402
from rrc.optim.hyper_search import build_states_for_config  # noqa: E402
from run_fine_grid import FIXED, GRID, REFERENCE, score  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
logger = logging.getLogger("heatmap")

DEFAULTS = dict(leak_rate=0.30, target_pre_activation_std=0.70, gate_temperature=1.00)


def draw(axis, matrix, n_rs, rhos, title, star):
    im = axis.imshow(matrix, cmap="viridis", aspect="auto", origin="lower")
    axis.set_xticks(range(len(rhos)), [f"{r:.2f}" for r in rhos])
    axis.set_yticks(range(len(n_rs)), [str(n) for n in n_rs])
    axis.set_xlabel("spectral radius  rho")
    axis.set_ylabel("reservoir size  N_r")
    axis.set_title(title, fontsize=11)
    for i in range(len(n_rs)):
        for j in range(len(rhos)):
            axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                      color="white" if matrix[i, j] < matrix.max() - 0.5 * np.ptp(matrix) else "black",
                      fontsize=8)
    si, sj = star
    axis.add_patch(plt.Rectangle((sj - 0.5, si - 0.5), 1, 1, fill=False,
                                 edgecolor="#e63946", lw=3))
    plt.colorbar(im, ax=axis, shrink=0.85, label="excess Sharpe")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N_r x rho val-vs-test heatmap.")
    parser.add_argument("--universe", default="sector11", choices=["etf10", "sector11"])
    parser.add_argument("--test-seeds", default="0,1,2")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT))
    parser.add_argument("--states-dir", required=True, help="Scratch dir used by run_fine_grid.")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    processed_dir = data_root / "data" / "processed"
    states_dir = Path(args.states_dir).expanduser().resolve()
    test_seeds = [int(s) for s in args.test_seeds.split(",")]

    spec = GRID[args.universe]
    k = spec["k"]
    n_rs = spec["n_reservoir"]
    rhos = spec["spectral_radius"]
    artifacts = load_universe_artifacts(processed_dir, args.universe)
    every_fold = list(folds_for(args.universe))

    grid = pd.read_csv(data_root / "results" / "runs" / "grid_fine"
                       / f"grid_fine_{args.universe}.csv")
    slice_ = grid[(np.isclose(grid.leak_rate, DEFAULTS["leak_rate"]))
                  & (np.isclose(grid.target_pre_activation_std, DEFAULTS["target_pre_activation_std"]))
                  & (np.isclose(grid.gate_temperature, DEFAULTS["gate_temperature"]))]

    val = np.full((len(n_rs), len(rhos)), np.nan)
    test = np.full((len(n_rs), len(rhos)), np.nan)
    logger.info("=== %s: default-nuisance slice (a %.2f, std %.2f, tau %.2f), %d test trainings ===",
                args.universe, DEFAULTS["leak_rate"], DEFAULTS["target_pre_activation_std"],
                DEFAULTS["gate_temperature"], len(n_rs) * len(rhos) * len(every_fold) * len(test_seeds))
    for i, nr in enumerate(n_rs):
        for j, rho in enumerate(rhos):
            row = slice_[(slice_.n_reservoir == nr) & (np.isclose(slice_.spectral_radius, rho))]
            val[i, j] = float(row.val_sharpe.iloc[0])
            params = {**DEFAULTS, "gate_temperature": 1.0, "n_reservoir": nr, "spectral_radius": rho}
            states = build_states_for_config(args.universe, params, FIXED, processed_dir, states_dir)
            packs = {f.name: build_fold_packs(artifacts, states, f) for f in every_fold}
            test[i, j], _ = score(packs, every_fold, k, DEFAULTS["gate_temperature"], test_seeds, "test")
        logger.info("  N_r %d done", nr)

    vi = np.unravel_index(np.nanargmax(val), val.shape)
    ti = np.unravel_index(np.nanargmax(test), test.shape)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    draw(axes[0], val, n_rs, rhos, "VALIDATION  (what the search maximises)\n"
                                   f"brightest: N_r {n_rs[vi[0]]}, rho {rhos[vi[1]]:.2f}", vi)
    draw(axes[1], test, n_rs, rhos, "TEST  (the honest, out-of-sample number)\n"
                                    f"brightest: N_r {n_rs[ti[0]]}, rho {rhos[ti[1]]:.2f}", ti)
    ref_nr, ref_rho, ref_test = REFERENCE[args.universe]
    figure.suptitle(
        f"{args.universe} K={k}: the validation-hot reservoir is NOT the test-hot reservoir\n"
        "red box = each panel's own best cell; they do not coincide - selecting N_r/rho on "
        f"validation misses the best test config (capacity ref test {ref_test:.3f})",
        fontsize=13, y=1.03)
    figure.tight_layout()
    out = data_root / "results" / "runs" / "grid_fine" / f"heatmap_val_vs_test_{args.universe}.png"
    figure.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(figure)
    logger.info("  val argmax %s = %.3f | test argmax %s = %.3f",
                (n_rs[vi[0]], rhos[vi[1]]), val[vi], (n_rs[ti[0]], rhos[ti[1]]), test[ti])
    logger.info("  wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
