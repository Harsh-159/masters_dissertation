# Decision-Focused Regime-Gated Reservoir Computing for Portfolio Construction

MSc Artificial Intelligence and Machine Learning dissertation, University of
Birmingham. Supervisor: Prof. Peter Tiňo.

---

## What the project is about

Most portfolio pipelines work in two steps: first predict what each asset will
do, then hand those predictions to an optimiser that picks the weights. The
model is therefore trained to be an accurate *forecaster*, which is not the same
thing as being a good *investor* — a model with the smaller forecast error can
still buy the wrong asset.

This project takes the other route. The whole system is trained end to end on
the quality of the decisions it actually makes: the loss is the risk-adjusted
return (excess Sharpe ratio) of the portfolio it really held, costs included.
Nothing in it ever forecasts a return.

On top of that it asks a specific question about **market regimes**. Most work
defines regimes with a statistical rule fitted beforehand, such as a hidden
Markov model or thresholds on the VIX. Here the regimes are instead *discovered
by the decision objective itself*. The question is whether regimes found that
way carve the market differently from statistical ones, and whether they perform
any better.

**The short answer.** They do carve it differently — the learned gate organises
on recent market *direction* while the statistical gates organise on
*volatility*, so the two schemes agree no more than chance — but it does not
pay: no learned strategy reliably beats a simple equal-weight portfolio, and a
firewalled holdout, touched once at the end, confirms that out of sample.

---

## The folders

| folder | what it is |
|---|---|
| **`regime-rc/`** | **The dissertation project.** All the code, experiments and results. This is the one that matters. |
| `report/` | The written dissertation (LaTeX). `Dissertation.tex` is the master file. |
| `portfolio-data/` | Early work. Downloads and cleans daily prices for a few asset universes. Superseded — `regime-rc` has its own data layer, though it reuses the cached downloads from here. |
| `portfolio-forecast/` | Early work. An XGBoost model that predicts each asset's next-day return and trades on it. Superseded — it is a predict-then-optimise design, which is the approach this dissertation argues against. |

The two `portfolio-*` folders are kept for the record only. **Do not build on
them.**

---

## `regime-rc` — the main project

### The idea in one paragraph

A **frozen** reservoir (a Simple Cycle Reservoir, with a random echo-state
network kept as a robustness check) turns the daily return series into a state
vector that encodes the recent history. A trainable **gating** network reads that
state and outputs a soft membership over `K` regimes. `K` trainable **expert**
heads read the same state and each propose a portfolio, and the final portfolio
is the blend of them weighted by the gate. Only the gate and the experts are
trained; the reservoir is never trained, so it can be computed once and cached.
The training signal is the risk-adjusted quality of the whole realised return
stream — never a per-period forecast error.

### Getting started

```bash
cd regime-rc
python -m venv .venv
./.venv/bin/pip install -e ".[dev]"
./.venv/bin/python -m pytest
```

Python 3.11+ (developed on 3.14). **No GPU, no cloud, no Docker** — because the
reservoir is frozen, every training run only touches small heads on cached
states and finishes in seconds to minutes on a laptop.

### Running things

`regime-rc/Makefile` is written to be a copy-paste cheat sheet: every command in
it is a full literal line, so you can either run `make <target>` or copy any line
straight into the terminal. It is ordered the way the pipeline runs:

```
data & reservoir  →  the ladder  →  searches  →  evaluation  →  holdout  →  figures
```

The three most useful:

```bash
cd regime-rc
./.venv/bin/python -m pytest                  # the test suite
./.venv/bin/python scripts/show_registry.py   # summary of every run ever made
./.venv/bin/python scripts/make_figures.py    # rebuild every figure from the registry
```

Everything else takes longer — the searches and the full evaluation run for
20–80 minutes each.

### Where to read more

Inside `regime-rc/`:

- **`README.md`** — the project's own readme, with the full layout and the two
  disciplines it enforces.
- **`FINDINGS.md`** — the research record. Every result, decision, reversal, bug
  and surprise, organised by kind rather than by date. **Start here.**
- **`HANDOFF.md`** — the full build history, session by session, including what
  was tried and rejected.
- **`SEARCH_FINDINGS.md`** — every hyperparameter search, in the order it was run.

---

## How the results are kept honest

Three habits run through the whole project.

**Everything is reproducible.** One call seeds Python, NumPy and PyTorch and
switches on deterministic algorithms, so a given seed always names the same
experiment. Every figure in the dissertation is rebuilt by a script from the
recorded results — none is made by hand or in a notebook.

**Nothing is edited after the fact.** Every run appends a row to an append-only
registry recording what was run and what came out. There is deliberately no way
to update or delete a row: a wrong result is corrected by adding a new one with
a note.

**The holdout was touched exactly once.** The last eighteen months of data were
removed from every file at load time, with a check that raises an error if any
run tries to cross that line. It was evaluated one time, at the very end, after
the configuration had been fixed — so nothing about the model could have been
chosen with knowledge of it.
