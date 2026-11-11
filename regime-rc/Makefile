# regime-rc - run commands.
#
# Two ways to use this file:
#   1.  make <target>            e.g.  make data
#   2.  open it and copy any command line straight into the terminal.
#
# Every command is written out in full (no variables, no loops) so it is
# copy-paste ready. Run everything from the regime-rc/ folder. The two
# universes (etf10, sector11) are always run separately.
#
# Pipeline order:  data -> reservoir -> ladder -> search -> evaluation
#                  -> (freeze) -> holdout

.PHONY: install test clean \
        data reservoir \
        baselines rung1 rung2 rung3 rung5 \
        comparison hyper grid k-search front-attribution \
        walkforward capacity inference verify joint lstm-search regimes frequency \
        regimes-signals vix-bestk fine-grid penalty-sweep \
        holdout \
        figures meeting-figures walkforward-figures capacity-figure grid-heatmap signal-gate-figure \
        registry diagnostics ladder

# ============================================================ setup ==========
install:
	./.venv/bin/pip install -e ".[dev]"

test:
	./.venv/bin/python -m pytest

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# ================================================ 1. data & reservoir =========
data:
	./.venv/bin/python scripts/run_data.py --universe etf10
	./.venv/bin/python scripts/run_data.py --universe sector11

reservoir:
	./.venv/bin/python scripts/run_reservoir.py --universe etf10
	./.venv/bin/python scripts/run_reservoir.py --universe sector11

# ========================================================= 2. the ladder =====
baselines:
	./.venv/bin/python scripts/run_baselines.py --universe etf10
	./.venv/bin/python scripts/run_baselines.py --universe sector11

rung1:
	./.venv/bin/python scripts/run_rung1.py --universe etf10
	./.venv/bin/python scripts/run_rung1.py --universe sector11

rung2:
	./.venv/bin/python scripts/run_rung2.py --universe etf10
	./.venv/bin/python scripts/run_rung2.py --universe sector11

rung3:
	./.venv/bin/python scripts/run_rung3.py --universe etf10
	./.venv/bin/python scripts/run_rung3.py --universe sector11

rung5:
	./.venv/bin/python scripts/run_rung5.py --universe etf10
	./.venv/bin/python scripts/run_rung5.py --universe sector11

# ================================================ 3. search (P8, LONG) ========
comparison:
	./.venv/bin/python scripts/run_comparison.py --universe etf10
	./.venv/bin/python scripts/run_comparison.py --universe sector11

hyper:
	./.venv/bin/python scripts/run_hyper_search.py --universe etf10
	./.venv/bin/python scripts/run_hyper_search.py --universe sector11

grid:
	./.venv/bin/python scripts/run_grid_search.py --universe etf10
	./.venv/bin/python scripts/run_grid_search.py --universe sector11

k-search:
	./.venv/bin/python scripts/run_k_search.py --universe etf10
	./.venv/bin/python scripts/run_k_search.py --universe sector11

front-attribution:
	./.venv/bin/python scripts/run_front_attribution.py --universe etf10
	./.venv/bin/python scripts/run_front_attribution.py --universe sector11

# ============================================= 4. full evaluation (P9, LONG) ==
walkforward:
	./.venv/bin/python scripts/run_walkforward.py --universe etf10
	./.venv/bin/python scripts/run_walkforward.py --universe sector11

capacity:
	./.venv/bin/python scripts/run_capacity_curve.py --universe etf10
	./.venv/bin/python scripts/run_capacity_curve.py --universe sector11

inference:
	./.venv/bin/python scripts/run_inference.py --universe etf10
	./.venv/bin/python scripts/run_inference.py --universe sector11

verify:
	./.venv/bin/python scripts/run_verify_frozen.py --universe etf10
	./.venv/bin/python scripts/run_verify_frozen.py --universe sector11

joint:
	./.venv/bin/python scripts/run_joint_search.py --universe etf10
	./.venv/bin/python scripts/run_joint_search.py --universe sector11

lstm-search:
	./.venv/bin/python scripts/run_lstm_search.py --universe etf10
	./.venv/bin/python scripts/run_lstm_search.py --universe sector11

regimes:
	./.venv/bin/python scripts/run_regime_interpretation.py --universe etf10
	./.venv/bin/python scripts/run_regime_interpretation.py --universe sector11

frequency:
	./.venv/bin/python scripts/run_frequency_analysis.py --universe etf10
	./.venv/bin/python scripts/run_frequency_analysis.py --universe sector11

# ================================================= 5. follow-up analyses ======
regimes-signals:
	./.venv/bin/python scripts/run_regime_interpretation_signals.py --universe etf10
	./.venv/bin/python scripts/run_regime_interpretation_signals.py --universe sector11

vix-bestk:
	./.venv/bin/python scripts/run_vix_on_bestk.py --universe sector11 --k 2 --n-reservoir 300 --spectral-radius 0.85
	./.venv/bin/python scripts/run_vix_on_bestk.py --universe etf10 --k 5 --n-reservoir 600 --spectral-radius 0.95

fine-grid:
	./.venv/bin/python scripts/run_fine_grid.py --universe etf10
	./.venv/bin/python scripts/run_fine_grid.py --universe sector11

penalty-sweep:
	./.venv/bin/python scripts/run_penalty_sweep.py --universe etf10

# ============================== 6. the holdout (run ONCE, after freezing) =====
holdout:
	./.venv/bin/python scripts/run_holdout.py --universe etf10
	./.venv/bin/python scripts/run_holdout.py --universe sector11

# ================================================= figures & reporting ========
figures:
	./.venv/bin/python scripts/make_figures.py

meeting-figures:
	./.venv/bin/python scripts/make_meeting_figures.py

walkforward-figures:
	./.venv/bin/python scripts/make_walkforward_figures.py --universe etf10
	./.venv/bin/python scripts/make_walkforward_figures.py --universe sector11

capacity-figure:
	./.venv/bin/python scripts/make_capacity_figure.py

grid-heatmap:
	./.venv/bin/python scripts/make_grid_fine_heatmap.py --universe sector11 --states-dir results/runs/grid_fine/_states

signal-gate-figure:
	./.venv/bin/python scripts/make_signal_gate_figure.py

registry:
	./.venv/bin/python scripts/show_registry.py

diagnostics:
	./.venv/bin/python scripts/make_data_diagnostics.py

ladder:
	./.venv/bin/python scripts/make_ladder_preview.py --universe etf10
	./.venv/bin/python scripts/make_ladder_preview.py --universe sector11
