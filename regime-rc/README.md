# regime-rc

**Decision-Focused Regime-Gated Reservoir Computing for Multi-Objective Portfolio Construction**

MSc AI and Machine Learning dissertation, University of Birmingham.
Supervisor: Prof. Peter Tino.

---

## The idea in one paragraph

A **frozen** reservoir (Simple Cycle Reservoir, with a random ESN as robustness
check) turns the daily return series into a state `x_t` encoding the whole
fading history. A trainable **gating** network reads `x_t` together with
market-level sentiment and, through a softmax, outputs a soft membership `π_t`
over `K` latent regimes. `K` trainable **expert** heads read `x_t` and each
propose a feasible portfolio; the target portfolio is the blend
`w_t = Σ_k π_{t,k} · w_t^(k)`. Only the gating and the experts are trained; the
reservoir is frozen and its hyperparameters are searched. The training signal is
the **risk-adjusted quality of the entire realised return stream** (excess
Sharpe, plus tail and load-balancing terms) — never a per-period forecast error.

**The contribution is the composition, not any ingredient:** regimes are
discovered *by the decision objective itself* on a frozen temporal feature
space. The falsifiable question is whether decision-defined regimes carve the
market differently from — and outperform — statistically-defined regimes (HMM,
VIX terciles) at equal capacity and equal information.

---

## Setup

```bash
python -m venv .venv
./.venv/bin/pip install -e ".[dev]"
make test
```

Python 3.11+ (developed on 3.14.3). **No GPU, no cloud, no Docker** — the
frozen-reservoir design means every training run touches only small heads on
cached states and completes in seconds to minutes on a laptop.

## Commands

| Command | What it does |
|---|---|
| `make test` | Run the test suite. This is the whole CI story this project needs. |
| `make figures` | Regenerate every figure of record from the frozen registry. |
| `make registry` | Print the registry summary and the deflated-Sharpe trial count. |
| `make clean` | Remove Python caches. |

## Layout

```
configs/          one YAML per experiment; sha256 of its canonical JSON = cfg_hash
scripts/          every result of record is produced by a script here, never a notebook
src/rrc/
  data/           loaders, FRED series, signals, availability matrix, calendar
  reservoir/      SCR, ESN, state cache, health diagnostics
  models/         gating, experts, losses
  backtest/       the closed-form differentiable ledger, costs, metrics
  baselines/      1/N, Markowitz, HMM gate, VIX gate
  optim/          training loop, per-date NSGA-II front, outer hyperparameter search
  eval/           walk-forward, statistics, regime slicing, the variant registry
  utils/          seeds, io
tests/            mirrors src; causality tests C1-C7 and engine tests B1-B7 live here
results/          registry.csv (versioned), runs/, figures/
notebooks/        diagnostics only, never results of record
```

---

## The two disciplines this repository enforces

### 1. The variant registry (`results/registry.csv`)

Append-only. Every run appends one row per (fold, split) before it writes any
other artifact. There is deliberately **no update or delete function** — a wrong
result is corrected by appending a new row with an explanatory `note`, never by
editing history.

Each row records **what was run** (`git_sha`, `cfg_hash`, universe, rung, arm,
seed, fold, split, `n_params`) and **what came out** (the Section 13 metrics).

`git_sha` is the commit at `HEAD` when the row was written, so it pins the
**base commit** rather than the exact bytes: research code is developed by
running, looking, tweaking and running again, so most rows were produced on a
working tree carrying uncommitted edits on top of that commit. An earlier
version appended a `-dirty` marker in that case; it was removed because it
applied to essentially every row and so distinguished nothing. The honest
statement is the one made here — the SHA locates a row in the project's history
and identifies its configuration exactly (`cfg_hash`), but reproducing a
historical row byte-for-byte may require the commit that followed it.

**The deflated-Sharpe trial count is pre-registered**, before any experiment
exists:

> `N_trials` = the number of **distinct `cfg_hash` values** in the registry with
> **`split == "val"`**, measured at the moment the DSR is computed.

Validation rows are the right population because selection happens on validation:
the count should measure how many configurations were *given a chance to be
selected*. Counting distinct hashes means re-running one configuration across
folds and seeds does not inflate it, while genuinely different configurations
always do. Writing this rule down first is what makes the deflation honest
rather than theatre.

### 2. Seed discipline (`src/rrc/utils/seeds.py`)

One call seeds Python, NumPy and PyTorch and switches on deterministic
algorithms. Results are reported as mean ± sd over head seeds `{0..4}`; that is
only meaningful if "seed 3" names the same experiment every time, which requires
*complete* seeding. The test suite checks each generator independently so a
partial-seeding bug cannot hide.

---

## Status

**All phases complete (P0–P11). All three gates (G1, G2, G3) passed. 344 tests
green.** The analysis is finished and the holdout has been evaluated; the
remaining work is the written report (`../report/`).
Registry: 28 distinct trial configurations across rungs {0, 1, 2, 3a, 3b, 3c, 4}
on both universes (`etf10`, `sector11`). Git tags: `g1-pass`, `g2-pass`,
`g3-pass`.

| Phase | State |
|---|---|
| **P0 — Scaffold** | ✅ repo skeleton, pinned environment, seed discipline, registry with pre-registered DSR rule, `make test` / `make figures` |
| **P1 — Data layer** | ✅ price/FRED/sentinel loaders, signal matrix, availability matrix; causality tests C1–C6 (C5 truncation-invariance the strongest) |
| **P2 — Backtest engine** | ✅ closed-form differentiable holding-period ledger (reference == vectorised to 1e-12); costs/turnover (A9); engine tests B1–B7 + C7; caught the all-cash Sharpe-degeneracy (E1) |
| **P3 — Reservoir + state cache** | ✅ deterministic SCR (ring, exact ρ, π-digit signs) + random-ESN robustness check (CCA 0.99); input scale calibrated to pre-tanh std ≈ 0.7; state cache; fixed the inverted memory-capacity probe-scale bug |
| **P4 — Rung 0 baselines** | ✅ 1/N, buy-and-hold SPY, inverse-vol, Ledoit–Wolf MV — the humility line, under the identical ledger/costs; stationary-bootstrap CIs |
| **P5 — Rung 1 → gate G1** | ✅ single linear head, full-span Sharpe training through the ledger; toy overfit + finite-difference gradient check; **C2 redefined as the conditioning edge**; etf10 edge ≈ 0 is the honest null |
| **P6 — Rung 2 → gate G2** | ✅ K-expert mixture + learned gate (noise anneal + balance loss); **split verdict**: etf10 experts specialise (val Sharpe 0.87→0.99), sector11 documented negative finding (one regime) |
| **P7 — Rungs 3a/3b/3c + 4 → gate G3** | ✅ sentiment routing (3a best → sentiment is a *routing* signal); statistical gates (filtered HMM, VIX terciles) at equal conditions via literal code reuse; agreement metrics — **RQ1 headline: AMI ≈ 0, decision-defined regimes carve the market differently** |
| **P8 — Rung 5 + hyperparameter search** | ✅ NSGA-II built, measured and **retired**; rung 5 redefined as exact cardinality; metric corrected to concatenated Sharpe; LSTM arm added |
| **P9 — Full evaluation** | ✅ walk-forward on both universes, capacity curve, inference layer, interpretability (F1), config frozen to the plan defaults |
| **P10 — Figures and deck** | ✅ paper figures, architecture diagrams, demo deck |
| **P11 — Holdout + write-up** | ✅ F1 closed (gate given VIX still tracks direction), 1,485-cell fine grid, **holdout evaluated once — frozen config ties 1/N**; report in progress |

Interim visualisations (growth-of-£10,000 ladders and per-universe dashboards)
live under `results/diagnostics/ladder/`; they are exploratory previews, not
results of record, and are superseded by P9's figures F4/F5.

The gates, not the dates, are the contract: G1 the mechanism works, G2 the
experts specialise or the null is documented, G3 the research question is
answerable in some direction. Everything after G3 is severable.
