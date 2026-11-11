# HANDOFF — complete state of the regime-rc build
_P0–P7 written 2026-07-18; P8 appended 2026-07-27; P9 appended 2026-08-07;
P10 appended 2026-08-14; **P11 appended 2026-08-29**._

> ## NEW SESSION: read §12 first, then §12.12 for what to do next.
>
> Three things changed at P11 and they override everything below:
>
> 1. **THE HOLDOUT IS SPENT** (§12.1). It was evaluated on 2026-08-17. It cannot
>    be run again. The result is in `results/runs/holdout/` and it **confirms the
>    null**: the frozen config lands on top of 1/N on both universes.
> 2. **DO NOT MENTION THE DEFLATED SHARPE RATIO** anywhere in the write-up
>    (§12.9). User's editorial decision, 2026-08-29. Argue the null through the
>    paired bootstrap, the Ledoit–Wolf test, the held-out verification and the
>    holdout instead. Several already-drafted report passages still contain DSR
>    references that need scrubbing.
> 3. **The analysis is FINISHED and the report is being written** (§12.10). The
>    deadline is **Tuesday 1 September 2026**. Do not start new experiments.

> **READ SECTION 6 BEFORE SECTION 5.** P9 corrected several P8 conclusions by measurement.
> Where §5 and §6 disagree, **§6 wins**. Specifically superseded: the K = 1
> selection, the "current best configuration" in §5.11, and the §5.9 claim that
> the mixture's failure is a universe limitation.

This document is the bridge between chat sessions. Read it together with the
three plan PDFs in `~/Desktop/dissertation/` (hierarchy: **Implementation Plan
v1.0** > **7-July Review** > **Architecture Specification**). It records what
has actually been built, every decision and deviation, every bug found, the
scientific findings so far, and exactly what remains.

---

## 0. Project identity

MSc AI/ML dissertation, University of Birmingham, supervisor Prof. Peter Tino,
window 7 July → mid-September 2026:
**"Decision-Focused Regime-Gated Reservoir Computing for Multi-Objective
Portfolio Construction."**

One-sentence contribution (from the Review §4): *regimes are discovered by the
decision objective itself on a frozen temporal feature space; the falsifiable
question is whether decision-defined regimes carve the market differently
from, and outperform, statistically-defined regimes at equal capacity and
equal information.*

## 1. Repository state

- **Repo:** `~/Desktop/AI/dissertation/` is the git repo (remote
  `git@git.cs.bham.ac.uk:projects-2025-26/hxy511.git`); the project lives in
  the `regime-rc/` subfolder. Two sibling folders `portfolio-data/` and
  `portfolio-forecast/` are **superseded early work — do not build on them**
  (their raw yfinance cache was reused; their XGBoost baseline is off-plan).
- **Tags:** `g1-pass`, `g2-pass`, `g3-pass` — all three plan gates passed.
  (There is **no** `scaffold` tag despite P0's checklist; cosmetic.)
  **Still no `config-freeze` or `results-freeze` tag** — see §6.9 for what
  blocks them.
- **Tests:** **344 green** (`make test`, ~20 s), zero skips when `data/` is
  present. hmmlearn convergence warnings in the rung-3 tests are benign EM
  chatter.
- **Registry:** `results/registry.csv`, **2960 rows**, rungs
  {0,1,2,3a,3b,3c,4,5,capacity,hyper,k-search}, **DSR trial count 449**
  (pre-registered rule: distinct `cfg_hash` with `split == "val"`). It went
  28 → 305 during P8's searches and 305 → 449 when P9's capacity curve
  registered 144 inner-search cells. Append-only; never edited, only appended.
  - ⚠️ **Open honesty issue.** P9's held-out verifications, the joint
    K × ρ searches and the LSTM searches were all run with `--dry-run`, so
    roughly **69 further trials are NOT in the registry** (5 verify + 24 joint
    + 40 LSTM). Under the pre-registered rule they should count: they were
    configurations given a chance to be selected. **The write-up must either
    register them or state the true count (~518) explicitly.** Do not quote 449
    as if it were complete. Deciding this is a P11 blocker.
- **Committed:** P8 and P9 are both committed on `main` (through
  `3fceade`), with later work uncommitted (see §6.10).
- **The `-dirty` SHA marker was removed** (2026-08-07). It applied to 2960 of
  2960 rows, so it distinguished nothing. `git_sha()` no longer computes it and
  the suffix was stripped from the registry (backup:
  `results/registry.csv.pre-dirty-strip.bak`). The weaker guarantee — the SHA
  pins the **base commit**, `cfg_hash` pins the configuration exactly — is now
  stated in the README instead of marked per row.
- **Venv:** `regime-rc/.venv`, Python 3.14.3. Pinned in `pyproject.toml`:
  torch 2.13.0 (CPU), numpy 2.5.1, pandas 3.0.3, pymoo 0.6.2, hmmlearn 0.3.3,
  scikit-learn 1.9.0, scipy 1.18.0, statsmodels 0.14.6, yfinance 1.5.1,
  pandas-datareader 0.11.1, openpyxl 3.1.5, matplotlib 3.11.1, PyYAML 6.0.3,
  pyarrow 25.0.0, pytest 9.1.1. **No GPU/cloud/Docker anywhere.**
- **Run everything from inside `regime-rc/`** (scripts use `./.venv/bin/python`
  via `make`; shell state does not persist between tool calls — always
  `cd ~/Desktop/AI/dissertation/regime-rc` first).

### Make targets
```
make test         # pytest suite (the whole CI story)
make data         # rebuild data artifacts (cache-first, offline after 1st run)
make diagnostics  # exploratory D1-D10 plots (NOT results of record)
make baselines    # rung 0
make rung1|rung2|rung3
make figures      # regenerate figures of record from registry/artifacts
make registry     # print registry summary + DSR trial count
```

### Data/universe facts
- **Universes:** `etf10` (10 ETFs, calendar 2006-02-01→2026-06-30, 5134 days)
  and `sector11` (9 original SPDRs from 1999-01-04 + XLRE 2015-10, XLC 2018-06
  via availability matrix; 6914 days). **DJIA dropped permanently** (user
  decision; free survivorship-free data does not exist — evidence in the old
  portfolio-data coverage reports: UTX/KFT/DWDP/WBA unretrievable, old GM 0%
  membership coverage).
- **Holdout:** 2025-01-01 → 2026-06-30, firewalled in code
  (`truncate_before_holdout` + `assert_no_holdout` in `eval/slicing.py`);
  untouched until the single Week-8 scripted run (P11).
- **Folds (eval/slicing.py, tested):** etf10 F01–F10 (val 2014–2023, test
  2015–2024); sector11 F01–F18 (val 2006–2023, test 2007–2024 → 2008 genuinely
  out-of-sample). 5-trading-day embargo removed from every train tail.
- **Signals matrix (P1):** vix, sentiment (SF Fed), term_slope, short_rate,
  credit_spread — publication-lag shift FIRST, trailing 252d z-score SECOND
  (order is load-bearing); d = 5 (plan assumed d = 6; parameter-count comments
  use our d = 5).

## 2. Phase-by-phase record (P0–P7 all complete)

### P0 — scaffold (tag `scaffold`)
`utils/seeds.py` (one call seeds python/numpy/torch + deterministic
algorithms), `utils/io.py` (canonical JSON → `cfg_hash`; atomic writes),
`eval/registry.py` (append-only CSV, file-locked, schema-guarded, **dtypes
from schema not inference** — the `rung` column is string so `"3a"` works).
**Pre-registered DSR rule** written before any experiment. Bug found: unanchored
`data/` in .gitignore silently untracked `src/rrc/data/` (fixed to `/data/`).

### P1 — data layer
`data/{loaders,fred,signals,availability,calendar}.py` + `scripts/run_data.py`.
Per-ticker full-history parquet cache (offline after first pull; rebuild 0.5 s).
r_f from **DGS3MO yield** with compound-daily conversion `(1+y/100)^(1/252)-1`
(A4 — not discount-basis DTB3). CASH column earns exactly r_f ⇒ all-cash
excess ≡ 0 (D5/E1). Reservoir-input universe excludes late entrants + cash
(R4). **Causality tests C1–C6** (C5 truncation-invariance is the strongest:
rebuild on data ≤ t is bit-identical). Bugs: `pd.NA` broke float casting
exactly on the sentiment-outage path (→ np.nan); make_figures needed
per-builder artifact guards.

### P2 — differentiable ledger
`backtest/{ledger,costs,metrics}.py`. Closed-form holding-period ledger in
float64 torch; **two implementations** (readable reference loop == vectorised
fast path to 1e-12, tested). E7 timing lives OUTSIDE the ledger in
`weekly_execution_schedule` (signal = last trading day of week, execute next
day). A9 pinned: cost = c·Σ|Δw| (full notional, 5 bp, cash leg included —
matches plan's C.1 exactly); one-way turnover = half that. Tests B1–B7 + C7.
**Critical bug caught (E1):** all-cash portfolio reported Sharpe 4.17 — float64
round-off (~1e-19 excess) in an unstabilised ratio; reporting Sharpe/Sortino
now carry the same ε=1e-6 denominator as the differentiable versions.
Verified on real data: B1 |cash−r_f|max 2.2e-16; B2 ledger==SPY to 2e-14.

### P3 — reservoir + state cache
`reservoir/{scr,esn,cache,diagnostics}.py` + `scripts/run_reservoir.py`.
Deterministic SCR (ring, exact spectral radius by construction, π-digit sign
pattern computed via Machin); random ESN as robustness check sharing the SAME
dynamics code. Input scale **calibrated** to pre-tanh std ≈ 0.7 on the first
60% of the sample (v ≈ 0.66 both universes; documented as a global reservoir
hyperparameter shared across folds — the plan's own convention). States cached
float32 npz keyed by config hash: etf10 4755 usable days, sector11 6538
(washout 250 + z-warm-up excluded; unusable rows are NaN not 0). SCR↔ESN
canonical correlation 0.987–0.989.
**Methodological fix:** memory-capacity curve initially INVERTED because I
probed at operating input scale; MC measures *linear* recoverability →
probe scale 0.05. F2 now rises 9.5→19.5 peaking at ρ=0.9 (α=1.0 reference
curve reaches 116); regression test pins the probe-scale dependence.

### P4 — rung 0 baselines
`eval/{slicing,stats}.py`, `baselines/{naive,markowitz}.py`,
`scripts/run_baselines.py`. Four arms: 1/N over the ACTIVE set, buy-and-hold
SPY (market line for BOTH universes), inverse-vol (63d), Ledoit–Wolf
mean-variance (252d, monthly refit, γ=5 fixed, **cash excluded from
covariance, held as residual budget** — E1). **Per-segment fresh start**
(each evaluated year starts in cash) exactly as models are evaluated — no
continuity advantage anywhere. Politis–Romano stationary bootstrap (block 21d,
B=2000; index-based so P9 can do paired resampling). Numbers (concatenated
test years, net): etf10 2015–24 — SPY 0.69 dominates, 1/N 0.37, inv-vol 0.28,
LW-MV 0.06; sector11 2007–24 — all cluster 0.48–0.51 but LW-MV maxDD −35.5%
vs −53% (cash-slack de-risking visible through the GFC). Legit-NaN guards
refined: inception-day returns and r_f's first row.

### P5 — rung 1 single head (tag `g1-pass`)
`models/{experts,losses}.py`, `optim/{fold_data,train_loop,toy}.py`,
`eval/causality.py`, `scripts/run_rung1.py`. Linear head → availability-masked
softmax (exp(−1e9) underflows to exact 0; cash never masked). Loss stack:
−excess Sharpe + λ_cvar·CVaR95 + λ_balance·CV²(Imp) (balance ≡ 0 for K=1).
Full-span batching only (A3). Toy protocol: finite-diff matches autograd to
6.9e-9; toy overfit Sharpe 38. F01 both universes × λ_cvar {0,1,5} × 5 seeds
all stable; early stopping picks epochs 3–6 (val peaks then decays).
**The C2 redefinition (important deviation, fully documented):** raw-Sharpe
shuffle-collapse FAILED on etf10 because shuffling preserves drift — a static
strategy keeps its Sharpe (buy-and-hold would be "flagged"). The correct C2
quantity is the **conditioning edge** = dynamic Sharpe − the strategy's own
static-counterpart Sharpe (average weights, same ledger, same world). Tests
prove power: a planted leak survives shuffling in every world and is caught; a
zero-weight static head is NOT accused. Result: etf10 edge −0.004 (the rung-1
head is essentially a static defensive tilt — the honest null for the thesis),
sector11 edge 0.032; both collapse under shuffling. λ_cvar showed no
cross-universe winner on F01 (etf10 liked 5.0, sector11 flat) → default 1.0
retained, **formal freeze deferred to P8**.

### P6 — rung 2 mixture (tag `g2-pass`)
`models/gating.py` (GatingNetwork with temperature + R2 noise annealed
0.3→0 over first half of max_epochs, eval always noise-free;
MixtureOfExperts blending w=Σπ_k·w^k; **K fully config-variable** — K=1
provably degenerates to rung 1, tested for K∈{1,2,4,7}), `train_mixture`,
`mixture_diagnostics` (importance shares, gate entropy, pairwise expert-weight
correlations, per-expert standalone vol/Sharpe — the ranking that feeds R1's
γ assignment at rung 5). Starvation governance: <5% share → one retrain at
λ_balance 0.05 (never needed in practice). **Split verdict:** etf10
SPECIALISED (median max pair-corr 0.473; credit/gold/equity experts visible in
F7; val Sharpe 0.99 vs rung-1's 0.87 — first evidence conditioning adds
value); sector11 documented NEGATIVE finding (corr 0.923, entropy ≈ log 3,
no gain — one regime in an all-equity universe; K-reduction decision recorded,
ladder continues at K=3 for arm comparability, K-sensitivity belongs to P10).

### P7 — thesis week (tag `g3-pass`)
`models/gating.py` gains `route_pack` protocol + `SignalRoutedMixture`
(3a/3b/3c) + `FixedRoutingMixture` (rung 4, π rides on the pack, no gate
params); `baselines/{hmm_gate,vix_gate}.py`; `eval/agreement.py`;
`scripts/run_rung3.py`; SegmentPack gains `signals` and `fixed_memberships`.
**R6 equal-conditions = literal code reuse** (one training loop for all six
arms). HMM: hmmlearn fits on train only; hand-written log-space forward filter
(A5/C.2) supplies causal probabilities — the two mandated tests pass
(bit-identical past under future perturbation; provable divergence from
smoothed predict_proba). **hmmlearn trap fixed:** with covariance_type="diag",
`covars_` returns (K,d,d) full matrices — `diagonal_variances()` extracts
properly. Degeneracy rule FIRED: sector11 F02 auto-reduced 3→2 states,
recorded. VIX gate: doubly-lagged causal trailing-252d terciles.
A8 exact parameter deltas tested (gate +K·d=15; experts +K·d·N=165/180).
Run: 6 arms × 3 folds (F01–F03) × 5 seeds × 2 universes (~180 trainings,
~35 min).

**P7 findings (results/runs/rung3/, F8):**
- **RQ1 headline: AMI ≈ 0 on every fold/universe/gate** (−0.03..+0.10).
  Learned regimes are FAST (dwell 1.6–2.2 wks, ~50–63% weekly switching);
  HMM states SLOW (7–52 wks). Decision divergence up to 0.93. Decision-defined
  regimes genuinely carve the market differently — the strongest-claim branch
  of A10, pending P9's performance leg.
- Learned regimes economically real: occupancy-conditioned Sharpe on sector11
  F01 test splits +3.9/+1.7/−1.3.
- **RQ3 direction:** 3a (signals→gate) is the best sentiment arm on both
  universes' val means and the only one ever beating plain rung 2 (sector11
  F02: 1.48 vs 1.22); 3b/3c help nowhere. Within seed-sd at 3 folds.
- All test CIs straddle zero at 3 folds (honest; GFC test years 2007–09 for
  sector). P9's full matrix decides performance.

## 3. Conventions locked in code (do not re-litigate)

- **E7 timing:** signal at close t (last trading day of week) → execute close
  t+1 → earn from t+2. Lag lives in `weekly_execution_schedule`, outside the
  ledger; C7 protects it.
- **A9 costs:** 5 bp × full traded notional (cash leg included, matching the
  plan's C.1); one-way turnover = ½Σ|Δw|; λ_turn = 0 (costs flow through r_d).
- **Excess everything** over the same DGS3MO-derived r_f cash earns; ε=1e-6
  stabilised denominators everywhere.
- **Per-segment fresh start** for every evaluated year, models and baselines
  alike.
- **Full-span batching, never mini-batches** (A3).
- **Registry before artifacts**; append-only; seeds are a column (not in
  cfg_hash); rung is a string.
- **Seed-0 convention** for bootstrap-CI streams and figure diagnostics.
- **CONCATENATED Sharpe is the metric** (added P8). Stitch each seed's fold
  segments into one continuous daily stream, take a single excess Sharpe,
  computed identically to the baselines and the literature. Never average
  per-fold Sharpes as a headline — on sector11 the two differ by 0.48.
  Per-year figures are a consistency breakdown reported beside it.
- **The two universes are NEVER run or aggregated together** (added P8).
  Every `scripts/run_*.py` now REQUIRES `--universe {etf10,sector11}`; there
  is no `all`. They are separate testbeds on different calendars (etf10 from
  2006, sector11 from 1999) and pooling them averages 2008 with 2015.
- **route_pack protocol:** every model variant implements
  `route_pack(pack, noise_scale)`; the single train loop and evaluators call
  only that (+ `weights_for_pack`). Rung-1 head keeps `(states, mask)` call.

## 4. Known deviations from the plan (all deliberate, all documented)

1. **C2 = conditioning edge**, not raw-Sharpe collapse (P5; the plan's literal
   form would flag buy-and-hold as leaking). Belongs in the methods chapter.
2. **Agreement metrics live in `eval/agreement.py`**, not appended to
   slicing.py as the plan's file map suggested.
3. **d = 5 signals** (plan's A8 example used d = 6).
4. **λ_cvar not yet frozen** — grid ran at rung 1, no cross-universe winner on
   F01; default 1.0 used since; formal freeze is a P8 deliverable.
5. **Rung-2 F01 re-trained inside run_rung3.py** for self-containment → its
   registry rows exist under two cfg_hashes (P6's and P7's identity dicts).
   Harmless (append-only; deterministic training gives identical metrics).
6. **DJIA dropped** (not stretch-gated — user decision, evidence recorded).
7. **`.gitignore` parked blocks:** the user keeps commented per-phase blocks at
   the end of `regime-rc/.gitignore` (P2–P5 present). **P6 and P7 blocks have
   NOT yet been added** — pending user request pattern.
8. **F5 addition promised:** a linear "£10,000 → £X" money-growth variant
   alongside the log equity curves at P9 (user requested).

### Added during the P8 session

9. **NSGA-II retired entirely** (§5.1) — the per-date front was built, measured
   at -1.2 Sharpe against plain cardinality, and deleted; `pymoo` removed. The
   outer search is random, not NSGA. Both are documented reversals of R1/E2 with
   measurements attached, not shortcuts.
10. **Rung 5 redefined** as exact cardinality on the trained blend (§5.2).
11. **The evaluation metric changed to concatenated Sharpe** (§5.3). This is the
    single most consequential change: it moves numbers by up to 0.48 and
    invalidates any sweep predating it.
12. **`return_annualisation`** added to the (now retired) front's selection rule
    to fix a scale degeneracy — kept in the record because it explains the
    all-cash rung-5 rows still in the registry.
13. **Four new loss knobs**, all defaulting to 0 and all measured-and-rejected:
    `weight_decay`, `lambda_div` (Spec 8.5's anti-redundancy — implemented for
    the first time), `lambda_drawdown`, and the earlier `lambda_turn`.
14. **K is no longer assumed to be 3** on sector11 (§5.7): K=1 wins in every
    round. G2's "keep K=3 for arm comparability" is now in tension with the
    evidence and needs an explicit decision in the write-up.
15. **`find_cached_states` rewritten** to identify the default reservoir by
    CONSTRUCTION rather than by globbing — the searches now leave ~70 caches per
    universe and the old glob-and-match became ambiguous.
16. **A bug I introduced and fixed**: the universe cleanup initially dropped the
    `universes =` assignment in five `run_*.py` scripts. Repaired; all scripts
    compile and CLI-smoke clean.

## 5. P8 — what was actually built (session of 2026-07-27)

P8 was executed and departs substantially from the plan. **Read §6 before
running anything**: several of the plan's P8 deliverables were built, measured,
and deliberately retired, and re-implementing them would repeat work that has
already been shown not to pay.

### 5.1 NSGA-II is GONE — both the front and the outer search

**The per-date front (R1) was built in full, measured, and deleted.** It is not
a stub or a simplification; `src/rrc/optim/nsga_front.py` and its tests were
removed and `pymoo` dropped from `pyproject.toml`. Three measurements retired it:

1. **A scale bug first made it degenerate.** The selection utility
   `mu^T w - gamma*CVaR` compares a DAILY EWMA mean (~2 bp) against a daily
   CVaR (~1.3%), so with gamma in {2,6,12} every risky point scored negative
   while cash scored exactly 0. The front locked into all-cash on date 1 and
   never left: turnover exactly 0.000, Sharpe 0.000, on every fold and seed.
   Annualising the return term fixed it (a documented deviation from R1).
2. **Once fixed it still lost badly.** `run_front_attribution.py` compares three
   schedules on ONE trained model, ONE segment, the SAME ledger:
   `A` the mixture blend, `B` the blend restricted to its top-K_card names
   (cardinality only, no front), `C` the NSGA front.

   | schedule | etf10 (val 2014-16) | sector11 (val 2006-08) |
   |---|---|---|
   | A mixture blend | +0.856 | +0.332 |
   | B blend + cardinality only | **+0.909** | **+0.342** |
   | C NSGA front | **-0.325** | **-0.231** |

   **Cardinality is free; the front's optimisation destroys ~1.2 Sharpe.** The
   cause is conceptual, not numerical: the front re-picks assets by an EWMA
   momentum estimate, discarding the trained system's own view. **R1 bolts a
   predict-then-optimise step onto the end of a decision-focused system** —
   precisely the pattern the thesis argues against — and it costs what the
   thesis predicts it should. That is a *result*, and belongs in the write-up.
3. **The combinatorics never justified it.** Choosing 5 of 10 assets is 252
   combinations, and once a subset is fixed the problem is convex. A stochastic
   population search was the wrong tool for a problem that small.

**The outer search is random search, not NSGA-II** (the plan's sanctioned
fallback). The module is now `src/rrc/optim/hyper_search.py` — it was misleadingly
named `nsga_hyper.py` and never used pymoo. NSGA's only advantage out here is a
Pareto front, which the outer search discards when it freezes one configuration.

### 5.2 Rung 5 is now EXACT CARDINALITY, and it has a cost

`src/rrc/optim/cardinality.py` + `scripts/run_rung5.py`. Rung 5 keeps the trained
blend's top-`K_card` names, renormalises, and runs the same ledger. No retraining,
no optimiser. **The 5 bp transaction cost is untouched** — `selection_cost` was a
regularisation knob inside the retired NSGA rule, not a cost.

Cardinality's price, measured properly (12 folds, 5 seeds, concatenated):

| K_card | 3 | 5 | 7 | 9 | unconstrained |
|---|---|---|---|---|---|
| Sharpe (K=1) | 0.546 | 0.588 | 0.637 | 0.663 | **0.670** |

**Cardinality COSTS about 0.08 Sharpe at K_card=5, rising to 0.12 at K_card=3**,
monotone in tightness. Note this **reverses** the earlier A/B reading above:
that was one seed, three folds, mean-of-folds. The 12-fold concatenated number
is the one to quote. Rung 5's honest finding is the quantified price of
realistic execution — which is exactly what it exists to measure.

### 5.3 THE METRIC WAS WRONG, and it was material

**Every sweep before this correction averaged per-fold Sharpes. That is not a
valid aggregation.** Sharpe is a ratio; averaging it across years implicitly
rebases wealth each January and weights a calm year equally with a crisis year.

On sector11 F01-F03 the same configuration scored:

    mean-of-folds  +0.429      concatenated  -0.053      difference 0.48

**The correct metric everywhere is the CONCATENATED one**: stitch each seed's
fold segments into one continuous daily stream and take a single excess Sharpe,
computed exactly as the baselines and the literature compute it. The strategy is
a *procedure* that runs continuously — wealth is not reset each January — and a
bespoke aggregation would also make every baseline comparison invalid.

Per-year Sharpes are reported **beside** it as a consistency breakdown, never as
the headline. Every P8 number below uses the concatenated metric; anything in
§2 predating it is on the old convention and should not be mixed with it.

### 5.4 The ladder comparison (`scripts/run_comparison.py`)

Seven arms, one protocol, one metric. sector11, **test** years 2007-2011,
3 seeds, concatenated:

| arm | Sharpe | ±sd | turnover | maxDD |
|---|---|---|---|---|
| **ours-single (rung 1)** | **0.231** | 0.036 | 16.07 | -0.511 |
| r0 Ledoit-Wolf MV | 0.191 | — | 3.70 | **-0.243** |
| r0 inverse-vol | 0.146 | — | 1.67 | -0.499 |
| ours-group (rung 2) | 0.135 | 0.097 | 9.14 | -0.496 |
| r0 1/N | 0.130 | — | 1.36 | -0.530 |
| **LSTM (trained encoder)** | 0.102 | 0.047 | 4.02 | -0.508 |
| r0 buy-and-hold SPY | 0.049 | — | 1.00 | -0.550 |

Per-year: 2007 +0.43, **2008 -0.97**, 2009 +0.96, 2010 +0.66, 2011 +1.32
(ours-single).

**The baselines are rebuilt on the same folds inside this script** — they had
originally been run over all 18 sector11 folds, so the old comparison was across
different decades. SPY gets its own two-asset panel (it is not a sector11 column).

**These Sharpes are low because the window is brutal, not because of a bug.**
Verified independently from raw prices: **SPY returned -1.4% over 2007-2011 with
26.6% volatility** (hand-computed excess Sharpe +0.076 vs the pipeline's +0.049,
the gap being the 5 bp entry cost). The same code on etf10's 2015-2024 window
gives SPY +0.682. **Fold choice, not code, drives the headline.**

### 5.5 The LSTM arm — decision D2 is now measured, not assumed

`src/rrc/models/recurrent.py` + `tests/test_recurrent.py`. An LSTM replaces the
reservoir and **nothing else**: same decision loss, same ledger, same folds,
seeds, costs, masking, and the same `LinearPortfolioHead` output. It drives the
shared training loop by presenting itself as a one-expert mixture via
`route_pack`. Windowed inputs (60 trailing days per signal day) — a stated
difference from the reservoir's fading memory, documented rather than hidden.

**It was pre-registered to lose, with reasons, before the run** (see the module
docstring — this is the discipline the plan applies to its evolve-the-heads
comparison). It lost: **0.102 vs ours-single's 0.231**, below three of four
baselines. D2 is now a measurement: the frozen reservoir wins *because* a few
hundred weekly decisions cannot train a recurrent encoder.

### 5.6 Hyperparameters — what was searched, and what it found

Random search (screening) then coarse-to-fine grid (refinement). The write-up
line is: *random search to identify which hyperparameters matter, then a focused
grid on the ones that do, stopped at the noise floor.*

**The first random search was invalid and is a cautionary tale worth keeping.**
It scored all 50 candidates at `eval_seed: 0` and froze the maximum at validation
Sharpe **0.817**. Re-run on five seeds that config scored **0.298**, and with
seed 0 excluded it was **indistinguishable from the default** (0.168 vs 0.165).
The entire advantage was one seed on one fold (F03 gave +0.635 at seed 0 where
every other seed gave ~-0.87). `eval_seeds` is now a list and fitness is
concatenated. **The pre-registered DSR discipline caught a false positive before
it reached the results** — a good methods-chapter example.

The second search (50 draws, 5 folds, 2 seeds, concatenated) was healthy
(max 0.450, 2nd 0.450 tied, mean 0.227 ± 0.070) but its winner beat the default
by only **+0.035, within one standard error**. Reservoir hyperparameters, searched
properly, find essentially nothing.

**Response surface across the 50 draws** (quartile spread vs seed noise 0.03-0.08):

| hyperparameter | spread | reading |
|---|---|---|
| N_reservoir | 0.186 → 0.281 | the only clear effect, ~2x noise |
| input target_std | 0.197 → 0.255 | marginal |
| spectral radius / leak rate / gate temp | 0.02-0.05 | below noise **in that sample** |

⚠️ **That last row is a trap and was corrected later.** At K=1, rho=0.70 costs
0.14 and alpha=0.90 costs 0.12. Their apparent flatness was a property of a
sample dominated by small-N_r, K=3 draws — *not* of the model. **"Flat in a
random sample" is not "unimportant."**

### 5.7 K and K_card — the largest finding of the session

`scripts/run_k_search.py`, three rounds of successive **replication** (K is
integer and already dense, so extra rounds buy seeds, not resolution). Findings
are appended to **`SEARCH_FINDINGS.md`** as each round completes — that file is
the detailed record and should be read alongside this one.

**K = 1 wins on sector11, in every round:**

| round | folds x seeds | K=1 | K=2 | K=3 | K=5 | K=6 |
|---|---|---|---|---|---|---|
| 1 screen | 8 x 2 | **0.611** | 0.544 | 0.541 | 0.525 | 0.528 |
| 2 replicate | 8 x 5 | **0.580** | 0.520 | 0.545 | — | — |
| 3 confirm | 12 x 5 | **0.670** | — | 0.639 | — | — |

**Why — and it is not a bug.** Params-per-decision and the train/val gap move
monotonically with K:

| K | params | params/decision | TRAIN | VAL | gap | best epoch |
|---|---|---|---|---|---|---|
| 1 | 3,913 | 6.5 | 0.482 | **0.487** | **-0.006** | 11.4 |
| 3 | 11,739 | 19.6 | 0.476 | 0.455 | +0.021 | 8.4 |
| 6 | 23,478 | 39.2 | 0.462 | **0.415** | **+0.047** | 6.6 |

**Train Sharpe does not improve with K.** The extra experts buy no fitting power
at all — they fit the same function with more degrees of freedom, so it is pure
variance. That is what a single-regime universe looks like from the inside.

**Three checks confirm it is not a training defect:**

* **Cross-universe.** On etf10 the experts DO specialise (pairwise correlation
  0.169 at K=2, 0.379 at K=3, vs sector11's 0.625-0.837), reproducing G2's split
  verdict on an independent run — so the mechanism works where structure exists.
  But every K is statistically tied there too (1.044-1.066).
* **Forcing specialisation makes it worse.** K=3 with lambda_div ∈ {0, 0.01,
  0.05, 0.2}: correlation falls monotonically 0.691 → 0.400 and Sharpe falls
  **with** it, 0.547 → 0.382. Never approaches K=1's 0.598.
* **Gate noise is not the cause.** Removing it changed 0.547 → 0.546.

### 5.8 Everything that was tried and FAILED (do not redo)

Five loss terms were implemented, measured on the concatenated metric, and all
select to **zero**:

| term | status | evidence |
|---|---|---|
| `lambda_cvar` | 0.0 best | 0.526 / 0.505 / 0.449 at 0 / 1 / 5 |
| `lambda_turn` | 0.0 best | 0.526 → 0.403 → 0.412 at 0 / 0.5 / 2.0 (turnover 9.2 → 1.8) |
| `lambda_div` | 0.0 (neutral) | 0.547 / 0.526 / 0.443 at 0 / 0.01 / 0.05 |
| `weight_decay` | 0.0 best | closes the train/val gap 0.84 → -0.11 and makes validation WORSE |
| `lambda_drawdown` | 0.0 best | worse Sharpe AND does not even reduce max drawdown |

**This is one coherent finding, not five disappointments: the train/validation
gap is NOT fixable overfitting — it is non-stationarity.** Patterns fitted on the
training fold genuinely do not transfer to the next year, so no loss term
evaluated on the training window can help. `weight_decay` is the cleanest proof:
it does exactly what capacity control should (gap 0.84 → -0.11) and validation
still falls.

**Also failed:** rung 3a on sector11 (macro signals into the gate — 0.265 vs
rung 2's 0.429; the gate can see VIX at 80 and still holds 9% cash into the
crash, because no expert offers a defensive option); **seed ensembling**
(averaging the five seeds' weight schedules gave +0.005 over the mean seed and
-0.036 against the best — the seeds are too tightly clustered, sd 0.031, for
averaging to cancel anything).

All terms remain implemented as **searchable knobs defaulting to 0**, documented
as measured-and-rejected. A negative result that cost compute is worth keeping.

### 5.9 The 2008 diagnosis

sector11 fold F03 (validation 2008) scores ~-0.86 on **every** configuration
tried. Diagnosed in detail:

* mean cash held through 2008: **0.095**, max 0.113. Monthly: Jan .08 … **Oct
  .08** … Dec .11 — *flat through the worst month in modern history*.
* All three experts hold ~9.5% cash — clones. Gate mass 0.31/0.35/0.34, uniform.
  The mixture collapses to one static ~90%-equity book and rides the crash to
  **-29.8%** (Sep-Dec alone **-26.6%**).
* Contrast fold F02 (val 2007, Sharpe +0.909), where expert 2 genuinely
  specialised as defensive (**40% cash**, peaking at 99%) and the gate used it.

**The bottleneck is not detection, it is the absence of a defensive action.**
sector11 is all-equity: the only defensive asset is cash, and across 1999-2007
holding cash mostly lost, so no expert ever learns to want it. Perfect crisis
detection routing between nine flavours of "long equities" still loses 30%.

**This is a universe limitation as much as a model one** — etf10 holds TLT, IEF
and GLD and handles stress far better. A defensible framing for the write-up:
*decision-focused regime gating can only express defensiveness that the asset
universe makes available.*

> ❌ **REFUTED IN P9. Do not use this framing.** The full etf10 walk-forward
> (§6.3) shows the models losing to buy-and-hold by a *wider* margin than on
> sector11 — 0.318 against SPY's 0.688, roughly half the money — and etf10
> *has* TLT, IEF and GLD. Giving the system a defensive action does not rescue
> it. The limitation is not the universe. Killing our own hypothesis with
> evidence is worth reporting; recycling it after it failed is not.

Also worth stating plainly: **the whole 2007-2011 window is hostile, not just
2008.** Without 2008, sector11 validates around +0.83; with it, +0.26.

### 5.10 A real bug still OPEN

**R2's gate-noise anneal never completes.** `annealed_noise_scale` targets half of
`max_epochs` (epoch 250), but early stopping fires at **epoch 7-30**. Every
mixture ever trained in this project therefore ran under **~90% of full gate
noise for its entire life** and never committed to a routing.

Removing it changed the K result by 0.001, so it is **not** why K=1 wins — but it
is wrong, it silently affects every rung-2/3/4 number in the registry, and the
fix is one line (anneal against actual epochs, or a horizon matched to observed
early stopping). **Not yet fixed.**

> ✅ **FIXED IN P9.** `MixtureTrainConfig.gate_noise_anneal_epochs` now defaults
> to `2 * patience`, so the schedule completes inside a run that early-stops.
> Two regression tests pin it (`tests/test_rung2.py`). Every P9 number was
> produced post-fix; every P8 number was not, so the two are not strictly
> comparable at rungs 2–4.

### 5.11 Where the configuration actually stands

> ⚠️ **SUPERSEDED BY §6.** The configuration below was validation-selected and
> was subsequently **rejected** on held-out folds: rho = 0.95 scored 0.526
> against the plain defaults' 0.534. The frozen configuration is now the plan's
> Appendix B defaults on **both** universes, and K = 1 no longer stands as the
> sector11 answer once each K is given its own reservoir search (§6.5).
> Read §6.4 and §6.5 before using anything in this subsection.

The reservoir grid was first run at **K=3**, then K=1 was selected — and the two
interact, so the first grid's answer was contaminated:

* `gate_temperature` is a **mathematical no-op at K=1** (softmax over a single
  logit is 1.0 at any temperature), so the +0.067 credited to tau_g=1.4 could
  not transfer;
* re-run at K=1, the optimum moved from (N_r 800, std 1.10) back to
  **essentially the defaults**: N_r 300, std 0.70, with only rho nudged
  0.90 → 0.95 (0.615 vs 0.598, seed spread 0.013).

**Current best sector11 configuration** (validation-selected, NOT verified,
NOT frozen):

    reservoir : N_r 300, rho 0.95, alpha 0.30, target_std 0.70, washout 250
    model     : K = 1  (single head; the gate is inert)
    loss      : lambda_cvar 0, lambda_turn 0, lambda_div 0, weight_decay 0
    execution : K_card unconstrained is best; cardinality costs ~0.08 at 5

**Nothing is frozen and there is no `config-freeze` tag.** Two previous
"winners" collapsed at exactly the verification step, so this one owes the same
check on held-out folds (**F13-F18 were never used in any selection**) before
anything is written to `configs/frozen/` and tagged.

## 6. P9 — what was actually built (session of 2026-08-07)

P9 completed the plan's full evaluation, built the entire inference layer, and
**corrected four P8 conclusions by measurement**. Read this section before
acting on anything in §5.

### 6.1 The inference layer, which did not exist

`src/rrc/eval/stats.py` held only a stationary-bootstrap CI for a single
series. It now also has, with 24 new tests in `tests/test_stats_p9.py`:

* **`deflated_sharpe_ratio`** — the DSR, plus `expected_max_sharpe_under_null`
  (the Bailey/Lopez de Prado Gumbel formula). Returns a dataclass carrying every
  input, so a result can never be quoted without its trial population.
* **`paired_sharpe_difference_interval`** — the stationary bootstrap on a
  *difference*, one index matrix applied to both streams so the day-level
  pairing survives. This is why P4 made the bootstrap return indices.
* **`ledoit_wolf_sharpe_test`** — the robust Sharpe-difference test, HAC lag
  ⌊1.5·T^(1/3)⌋, Bartlett kernel. Delta method on the four uncentred moments.
* **`annualise_sharpe` / `deannualise_sharpe`** — the √252 conversion is now
  always a named call. The registry stores ANNUALISED Sharpes; the DSR algebra
  needs PER-DAY ones with T in days. Getting this wrong produces a
  plausible-looking answer wrong by √252, and it has its own regression test.

**Bug found and fixed by a test:** `ledoit_wolf_sharpe_test` returned p = 0
(infinitely significant) when comparing a stream against itself, because the HAC
variance collapses to exactly zero and 0/0 fell through to `inf`. Now returns
p = 1.

### 6.2 The DSR result, and the one judgement call it needs

The pre-registered rule fixes **N** (distinct `cfg_hash`, `split == "val"`). It
says **nothing about sigma**, the dispersion across trials — and sigma dominates,
because the benchmark is proportional to it but only logarithmic in N.

Two estimators are computed and BOTH are reported (`scripts/run_inference.py`):

| estimator | sigma | benchmark |
|---|---|---|
| pooled (all trials, both universes, all rungs) | 0.365 | **1.058** |
| coherent, sector11 (search rungs, live rows, one metric era) | 0.206 | 0.598 |
| coherent, etf10 | 0.117 | 0.340 |

The coherent estimator's three exclusions are each justified by a convention
locked BEFORE this calculation existed: §3's "universes are never aggregated",
§9's "filter rung-5 rows with ann_turnover > 0", and §5.3's "mean-of-folds is an
invalid aggregation". That is what makes them principled rather than convenient.

**Result: no learned arm clears 0.95 on either universe under either
estimator.** Best is 0.476. The pre-specified baselines pass because they were
never searched — deflating them would be a category error, so they are scored at
zero dispersion (which reduces the DSR to the probabilistic Sharpe ratio).

**Write-up instruction: report the pooled number FIRST**, then argue for the
coherent one. A reader who computes the sceptical number and finds it already in
your table will trust the rest of the chapter.

### 6.3 The full walk-forward, both universes

`scripts/run_walkforward.py` — 13 arms × all folds × 5 seeds, saving each arm's
stitched daily stream so figures and statistics regenerate in seconds without
retraining. `scripts/make_walkforward_figures.py` draws F4/F5/F5-money.

**sector11 (18 folds, 2007–2024) and etf10 (10 folds, 2015–2024), test, net:**

| arm | sector11 | etf10 |
|---|---|---|
| r0 1/N | 0.485 | 0.374 |
| r0 buy-and-hold SPY | 0.510 | **0.688** |
| r0 inverse-vol | **0.511** | 0.278 |
| r0 Ledoit-Wolf MV | 0.479 | 0.058 |
| LSTM | 0.378 | 0.299 |
| ours K=1 (rung 1) | 0.414 | 0.296 |
| ours K=3 (rung 2) | 0.461 | 0.237 |
| 3a sent-gate | 0.488 | 0.280 |
| 3b sent-experts | 0.469 | 0.272 |
| 3c sent-both | 0.500 | 0.264 |
| **ours K=1 + sentiment** | **0.507** | **0.318** |
| r4 HMM gate | 0.418 | 0.312 |
| r4 VIX gate | 0.408 | 0.232 |

**No learned arm beats a baseline on either universe.** On sector11 the best
ties inverse-vol at 10× the turnover; on etf10 SPY roughly doubles it
(£34,059 vs £15,545 from £10,000). **Zero of 24 adjacent-rung comparisons are
significant** — every paired-bootstrap CI straddles zero, Ledoit-Wolf p from
0.15 to 0.98.

**Reframe for etf10, and it matters:** 1/N scores 0.374 there and inverse-vol
0.278. *Every* diversified portfolio lost to SPY over 2015–2024. The models sit
next to 1/N because that is what they learned to be. Not uniquely broken.

**Fold overlap, quantified.** Walk-forward folds overlap by construction (year Y
is fold i's validation and fold i−1's test), so 7 of 18 sector11 and 4 of 10
etf10 test years were visible to hyperparameter selection. Model parameters are
unaffected. Restricted to years no search ever saw, the learned arms shift
+0.02 to +0.06 and the **baselines shift more** (SPY +0.21) — the overlap does
not flatter us. Volunteer this in the write-up.

### 6.4 Six selections, none transferred — and the frozen configuration

Every searched winner was re-scored on folds excluded from all selection
(sector11 F13–F18; etf10 F06–F10). `scripts/run_verify_frozen.py`,
`run_joint_search.py`, `run_lstm_search.py` — all universe-aware.

| # | universe | encoder | on validation | held out |
|---|---|---|---|---|
| 1 | sector11 | reservoir | 0.817 | 0.298 across seeds |
| 2 | sector11 | reservoir | won at K=3 | died on the move to K=1 |
| 3 | sector11 | reservoir | 0.603 | 0.526 (worst of three) |
| 4 | sector11 | reservoir | 0.582 | 0.428 (worst of three) |
| 5 | etf10 | reservoir | 0.464 | 0.783 vs defaults 0.809 |
| 6 | both | **LSTM** | 1.011 / 1.068 | 0.738 / 0.473 |

**Decisive measurement:** out of sample the spread ACROSS configurations (0.064)
is smaller than the spread between two SEEDS of one configuration (0.085).

**So the frozen configuration is the plan's Appendix B defaults on BOTH
universes** — the only choice requiring no look at any data, and therefore the
only one carrying no selection bias:

    reservoir : N_r 300, rho 0.90, alpha 0.30, target_std 0.70, washout 250
    gate      : tau_g 1.0
    loss      : all zero (lambda_balance 0.01 only when K > 1)
    sector11  : K = 1     etf10 : K = 3 (plan default; no K selection was made)

`configs/frozen/{sector11,etf10}.yaml` and `configs/frozen/rung5_frozen.yaml`
were all rewritten — they previously held unverified search winners and, in
rung5's case, configuration for the deleted NSGA module. Each file now documents
why. `configs/rung5.yaml` and `configs/comparison.yaml` also carried loss terms
measured to select to zero; corrected.

### 6.5 The capacity curve — this CORRECTS the K = 1 result

`scripts/run_capacity_curve.py`. K = 1…8, and critically **the reservoir is
searched INSIDE each K** (9 cells: N_r × rho), then the winner is scored on the
full walk-forward TEST split. The earlier K search compared every K at ONE
reservoir, tuned for one of them.

**sector11 test:** K=1 **0.403** (worst) · K=2 **0.505** (best) · K=3 0.474 ·
K=4 0.485 · K=5 0.459 · K=6 0.439 · K=7 0.455 · K=8 0.454.
Resolved beyond two seed spreads against K=1: **K=2 (+0.102), K=5, K=7, K=8**.

**Three findings:**

1. **K = 1 is the WORST capacity on test, not the best.** The P8 selection was
   an artefact of a fixed reservoir.
2. **On sector11 validation and test ANTI-CORRELATE, r = −0.477.** K=4 has the
   best validation of all eight and mid-table test; K=2 has near-worst
   validation and the best test. Selecting K on validation points the WRONG WAY.
   This is the single sharpest evidence for the transfer-failure result. It does
   NOT replicate on etf10 (r = +0.678) — a sector11 finding, not a law.
3. **K and the reservoir interact:** searched width climbs with K.

**But the frozen config STAYS at K = 1.** Re-selecting on a test-set curve is
exactly the bias six held-out checks were run to avoid. Report the curve as a
finding; do not act on it, and say so.

⚠️ etf10 seed spreads reach 0.155 at 3 seeds and are unreliable — a re-check of
K=7 at 8 seeds moved it 0.215 → 0.276 and sd 0.007 → 0.088. **Re-run the curve
at 5+ seeds before it goes in the report** (~40 min). sector11's spreads
(0.002–0.042) are trustworthy.

### 6.6 The LSTM comparison, now fair (D2)

The ladder had **238 reservoir-side search trials against 0 for the LSTM**.
`scripts/run_lstm_search.py` fixes that: hidden_size × lookback (16 cells), then
layers/dropout/learning-rate one at a time, then the held-out check.
`num_layers` and `dropout` did not exist as knobs and were added.

**Two findings, replicated on both universes:**

* **Capacity matters 6.8× (sector11) and 15.5× (etf10) more than memory
  horizon.** Averaged over lookback, hidden 16→128 swings +0.456 / +0.347
  against median seed spreads of 0.036 / 0.090. **This contradicts the argument
  that fixed hidden = 32.**
* **The memory horizon barely matters** — lookback 20→250 swings only +0.067 /
  +0.022, and shorter is if anything better. So the 60-day window was never the
  handicap. This answers "did you cripple the LSTM?" with a number, and it is
  corroborated by the reservoir's measured memory capacity of **19.53 with 69%
  of it inside 60 days**.

Neither champion transferred. D2 survives, and the comparison is now defensible
— but **state as a limitation** that on the folds a practitioner would have
searched, hidden = 128 beats the ladder's hidden = 32 by ~0.5.

### 6.7 Interpretability — the strongest new material

**Level 1 (`scripts/run_regime_interpretation.py`, figure F9).** G3 established
AMI ≈ 0 — the carvings disagree. It never said what either scheme IS. Method:
each week already carries a regime label; describe the market that week with
causal state variables; compute eta squared (fraction of each variable explained
by the labelling) and the standardised mean per regime.

**Finding, in all six runs across both universes:**

| eta squared, % | learned | HMM | VIX |
|---|---|---|---|
| realised volatility (sector11) | 0.8 | **44.3** | 18.9 |
| **recent direction, 5d** | **12.5** | 0.2 | 4.5 |
| realised volatility (etf10) | 0.3 | **25.2** | 11.3 |
| **recent direction, 5d** | **24.8** | 0.1 | 3.2 |

**The statistical gates are RISK detectors; the learned gate is a DIRECTION
detector, and each is nearly blind to the other's variable.** The learned
regimes mean "the market just rose / sideways / just fell"; the HMM's mean
"calm / stressed / crisis" with direction flat at ±0.03σ.

**This gives AMI ≈ 0 a mechanism** and predicts the dwell times: volatility
persists (HMM 12.9 wks) while 5-day direction flips constantly (learned 1.8
wks). Corroborated by the two RISK detectors agreeing with each other (AMI
+0.093 on sector11) more than either agrees with the learned gate (≈0.00).

**Caveat that must be stated:** this runs at K = 3, but sector11's frozen
configuration is K = 1, where the gate is mathematically inert and there are NO
regimes to interpret.

**Level 2 (`scripts/run_frequency_analysis.py`) — a pre-registered prediction
that FAILED.** The cycle matrix stays circulant under leaking
(A = (1−α)I + αρP), so the ring modes are exactly the Fourier basis, each with a
known memory τ_k = −1/ln|λ_k| spanning **1.2 to 32.8 days**. Prediction, stated
before the run: the gate should concentrate on short-memory modes.

**It does not.** The trained gate's spectrum is indistinguishable from random
initialisation — weighted-average memory 7.7 → 8.5 days (sector11) and
8.1 → 8.4 (etf10); on etf10 the largest per-mode shift (0.0032) is smaller than
a flat spectrum's own bin height (0.0066). The slow-weighted effective read
(~12 days) comes from the STATE spectrum, not the readout.

⚠️ **A 2-fold smoke test looked dramatic (25.9% vs 45.6% fast) and evaporated on
18 folds.** A second instance of a small sample giving a confident wrong answer.
Worth keeping beside the DSR false positive. ✅ **Written up in
SEARCH_FINDINGS.md (2026-08-07), including the falsified pre-registration.**

### 6.8 Everything else P9 fixed or checked

* **Gate-noise anneal** (§5.10) fixed; see the note there.
* **C8, a new end-to-end causality test:** a model trained on fold F03
  (validation 2008) holds **exactly 0.0** — not "small" — in XLRE and XLC, and
  the book still sums to 1.0. This makes "adding two sectors later cannot change
  earlier results" executable rather than argued. Note the load-bearing
  assumption it protects: `ledger_panel = returns_simple.fillna(0.0)` is safe
  only because the mask is exact.
* **Stream-name manifest.** Path-safe filenames are lossy, so the money table
  silently failed to join the summary table. `arms.json` now carries the exact
  mapping.
* **Ladder ordering and neutral arm names.** `LADDER_ORDER` drives tables and
  figure legends. `OPTIMAL_K` was renamed `K_SINGLE`/`K_MIXTURE` because calling
  K = 1 "ours-optimal" on etf10 imported a sector11 selection that no etf10
  evidence supports.
* **Meeting figures** (`scripts/make_meeting_figures.py` →
  `results/meeting_figures/`): M1 ladder, M2 orthogonality (all three schemes,
  all three pairings, AMI computed not hardcoded), M3 what each scheme tracks,
  M4 capacity curve, M6 growth of £10,000 (one panel per universe).

### 6.9 What blocks `config-freeze` and `results-freeze`

1. ~~Dirty SHAs~~ — **resolved 2026-08-07** by removing the marker (see §1).
2. **~69 P9 trials are unregistered** (the `--dry-run` searches; see §1). Decide:
   register them, or state the true count (~518) in the write-up.
3. **The capacity curve needs 5+ seeds on etf10** before it is quoted (§6.5).

---

## 7. What remains (P10–P11), concretely

**Calendar reality.** Today is **2026-08-07**. Deck **14 Aug** (7 days), demo
**17–28 Aug**, report **1 Sept** (25 days), **max 40 pages excluding
references**. The analysis is finished; everything left is housekeeping and
writing, and the binding constraint is now the deck.

### The three findings to write up

**F1 (the contribution).** Decision-defined regimes are orthogonal to
statistical ones BECAUSE they measure a different quantity: risk versus
direction (§6.7). Answers RQ1 with a mechanism, not just a number. Independent
of the performance null.

**F2 (the null).** The carving does not pay — 13 arms, two universes, 28
fold-years, zero significant differences, nothing clears deflation (§6.3). And
we refuted our own "no defensive asset" explanation.

**F3 (the methodological finding).** Hyperparameter selection does not transfer
on this problem, for either encoder (§6.4), with the capacity curve's
anti-correlation as the sharpest instance (§6.5).

Supporting: D2 settled fairly (§6.6); capacity not signal is the binding
constraint (the five loss-term nulls, §5.8).

**Demote to appendix:** the Level-2 frequency null, the K × ρ interaction (does
not replicate on etf10), the NSGA retirement, the mean-of-folds correction.

### Limitations to state explicitly

Single-path purged walk-forward, not CPCV (no distribution of paths, no PBO) ·
the LSTM is stateless across windows · fold overlap (7/18 and 4/10 test years
visible to selection — with the clean-years table showing it does not flatter) ·
√252 assumes independence, measured ρ₁ ≈ −0.10 so it is mildly conservative, and
inference uses HAC · Level-1 interpretability runs at K = 3 while the frozen
sector11 config is K = 1.

### P11 — the single holdout touch

**The holdout is untouched and provably so:** zero rows with `split == "holdout"`
in 2960, and `truncate_before_holdout` + `assert_no_holdout` remove 2025-01-01 →
2026-06-30 at load time, so those dates are absent from memory in every run.

Do it **after** the deck and demo, around **20–25 August**: freeze first, then
one scripted run writing `split="holdout"` rows, then the final DSR against the
count at that moment, quoted verbatim. The plan's warning stands — if a bug
surfaces afterwards, report both numbers and the bug; silently redoing it is the
one unforgivable move.

Expect it to be **limited as evidence** (≈375 trading days, huge standard error)
but **high as credibility**.

---

### 6.10 Documents added after the P9 runs (2026-08-07)

* **`FINDINGS.md` (NEW, 539 lines) — read this before writing anything.** The
  complete research record organised by KIND of finding rather than by session:
  results (F1–F9), 27 design decisions with their reasoning, 6 reversals,
  10 bugs with what each would have cost, 8 methodological lessons, open
  caveats. It supersedes nothing but collects what was scattered across this
  file, `SEARCH_FINDINGS.md` and ~14 module docstrings.
* **`SEARCH_FINDINGS.md`** gained the Level-2 frequency section.
* **The `-dirty` marker was removed** (§1).

**Uncommitted at the time of writing:** `FINDINGS.md`, `HANDOFF.md`,
`README.md`, `SEARCH_FINDINGS.md`, `results/registry.csv`,
`src/rrc/eval/registry.py`, `results/meeting_figures/`, and the P9 scripts
`run_capacity_curve.py`, `make_capacity_figure.py`, `make_meeting_figures.py`,
`run_frequency_analysis.py`.

---

## 8. START HERE (superseded by §11.7 — kept as the P9 record)

0. **Read `FINDINGS.md` first** — it is the curated research record and the
   skeleton the methods chapter writes from.
1. **Then §6 of this file.** It corrects §5 in four places.
2. ~~**Draft the deck.**~~ Done at P10 — see §11.4. Figures are in
   `results/analysis_figures/`, **not** `meeting_figures/`, which does not exist.
3. **Tag `config-freeze`** — the configuration is decided and defensible. Hold
   **`results-freeze`** until the capacity curve is re-run at 5+ seeds and the
   trial-count decision is made (§6.9). Both need the current work committed
   first; tagging `3fceade` would point at a tree without it.
4. **Write.** Three findings, two supports, 40 pages.
5. **Holdout last**, ~20–25 Aug.

**Do NOT spend time on** (all measured, all negative): more loss terms, NSGA in
any form, seed ensembling, rung 3a on sector11, forcing expert specialisation,
finer hyperparameter grids, or volatility targeting (considered and dropped —
it competes with writing time and the null is already evidenced six ways).

**Two runs worth doing only if the writing runs ahead:** the capacity curve at
5+ seeds (§6.5), and the ladder's LSTM arm at hidden = 128 as a reported
sensitivity check (§6.6).

---

## 9. Files changed in the P8 session (historical record)

**New source and tests**
```
src/rrc/optim/cardinality.py      exact cardinality (replaces nsga_front)
src/rrc/optim/hyper_search.py     renamed from nsga_hyper; multi-seed, concatenated
src/rrc/models/recurrent.py       the LSTM comparison arm
tests/test_cardinality.py         10 tests
tests/test_recurrent.py            7 tests
tests/test_turnover_loss.py        lambda_turn
```
**New scripts and configs**
```
scripts/run_comparison.py         the 7-arm ladder table (the main deliverable)
scripts/run_rung5.py              cardinality execution layer
scripts/run_front_attribution.py  the A/B diagnostic that retired NSGA
scripts/run_hyper_search.py       random search driver
scripts/run_grid_search.py        coarse-to-fine grid
scripts/run_k_search.py           K x K_card, three rounds, writes SEARCH_FINDINGS.md
configs/{comparison,rung5,hyper,grid,grid_k1,k_search}.yaml
SEARCH_FINDINGS.md                the running search record — READ THIS
```
**Deleted**: `src/rrc/optim/nsga_front.py`, `tests/test_nsga_front.py`;
`pymoo` removed from `pyproject.toml`.

**Modified**: `losses.py` (divergence + drawdown terms), `train_loop.py`
(`weight_decay`, `lambda_div`, `lambda_drawdown`), `fold_data.py`
(`input_windows` for the LSTM), `run_rung1.py` (`find_cached_states`),
`registry.py` + `seeds.py` (stale NSGA references), `Makefile` (per-universe
targets), all `run_*.py` (`--universe` required), `.gitignore`, `tests/test_rung2.py`.

## 10. Session gotchas

- Always `cd ~/Desktop/AI/dissertation/regime-rc` (shell resets per call);
  use `./.venv/bin/python` or `make`.
- **Every `run_*.py` now REQUIRES `--universe etf10|sector11`.** There is no
  `all` and the scripts error without it. `make` targets call each universe
  explicitly.
- **`data/states/` now holds ~70 caches per universe** from the searches, not
  one. `find_cached_states` resolves the DEFAULT config by construction, so
  it is safe — but do not assume a glob returns a single file.
- **Long runs**: the searches take 20-80 min each. Launch with `nohup ... &`
  and keep the Mac awake (`caffeinate -is -w $(pgrep -f <script>)`); closing
  the lid suspends them, closing the terminal does not.
- **Smoke tests pollute the registry.** `run_rung5.py` has `--dry-run` for
  this; the search scripts do too. An early rung-5 smoke added ~66 throwaway
  rows before the flag existed.
- **The all-cash rung-5 rows** (turnover exactly 0.000) are the pre-fix NSGA
  front, kept because the registry is append-only. Filter them out with
  `ann_turnover > 0` when reading rung-5 results.
- Long runs (`run_rung3.py` ~35 min; P8 search longer): launch with
  `run_in_background`, monitor the output file; piping through `tail` masks
  exit codes.
- The registry is append-only — re-runs add rows; that is correct behaviour.
- `results/` figures/runs are gitignored EXCEPT `registry.csv` (versioned
  audit trail). Data and states are regenerable (`make data`,
  `scripts/run_reservoir.py`).
- hmmlearn: use `diagonal_variances()`, never reshape `covars_`; "Model is
  not converging" warnings on toys are benign.
- pandas 3.0 quirks: no `pd.NA` into float pipelines; DatetimeIndex unit
  mismatches (ns/us/ms) — the state cache canonicalises to ns.
- `conditioning_edge` works for rung-1 heads and base mixtures, but NOT for
  SignalRoutedMixture via the (states, mask) path — extend if C2 is ever run
  on rung-3 arms.
- User preferences: build phase → then explain in simple, detailed words;
  keep code readable even if longer; K stays a config knob; per-phase
  commented "parked" blocks appended to .gitignore on request (P6/P7 pending).

### Added in P9

- **The worktree trap.** P9 ran in a git worktree at
  `.claude/worktrees/dissertation-next-steps-e9a85c/`, on its own branch, while
  RESULTS were written to the main checkout via `--data-root`. Code appeared to
  be "missing" from `~/Desktop/AI/dissertation/regime-rc/` for most of the
  session. If code seems absent, check `git branch --show-current` and whether
  you are in a worktree.
- **`--dry-run` leaves no registry rows.** Convenient while orienting, but P9
  used it on the verification, joint and LSTM searches, which is why ~69 trials
  are unregistered (§1). Decide deliberately, not by default.
- **Seed spreads from 3 seeds are unreliable.** etf10 K=7 measured sd 0.007 at
  3 seeds and 0.088 at 8. A 3-sample sd has ~50% coefficient of variation. Do
  not read individual sd values on etf10; use the typical spread as the floor.
- **The reservoir is deterministic**, so seed variation comes ONLY from the
  trainable head and gate initialisation. Reservoir width cannot drive seed
  spread — useful when a spread looks anomalous.
- **Long runs**: `nohup caffeinate -is <script> &`. Do not write a wait loop
  whose own command line contains the script name — `pgrep -f` matches the
  waiter itself and it never exits.
- **`scipy` is now a direct dependency** of `eval/stats.py` (norm, skew,
  kurtosis). It was already pinned.

---

## 11. P10 — housekeeping, figures and the deck (session of 2026-08-14)

No analysis was re-run. No result changed. This session was the write-up
infrastructure: the two superseded folders were simplified, the results figures
were rebuilt to publication standard, and the demo deck was produced.

### 11.1 Three numeric discrepancies that must be resolved before the report

These are the only findings of this session that touch the science. **Resolve
them before the results chapter is written**, because each appears in
`FINDINGS.md` as evidence.

1. **Learned-gate direction η² disagrees with the data.** `FINDINGS.md` F1 and
   §6.7 quote **12.5% (sector11) / 24.8% (etf10)**. Recomputed straight from
   `results/runs/regimes/separation_*.csv`, the values are **20.9% / 29.6%** —
   and the figures plot the CSV, so the docs and every figure now disagree.
   The direction of the finding is unaffected (HMM is 0.2% / 0.1% either way),
   but the number in the text must match. Find out which run produced 12.5.
2. **HMM dwell time.** Docs say **12.9 weeks**; computed from
   `state_and_labels_sector11.parquet` over the full test period it is
   **13.1 weeks**. Learned (1.8) and VIX (3.8) match. Likely a different
   window; pick one and use it everywhere.
3. **AMI values agree exactly** (+0.004 / +0.008 / +0.093), so nothing to do
   there — noted so the next session does not re-check it.

### 11.2 New figure pipeline — `scripts/make_paper_figures.py`

One script, six figures, each written as **PDF (vector, for the report)** and
**PNG at 300 dpi (for slides)** into `results/analysis_figures/`:

| file | replaces | content |
|---|---|---|
| `fig_ladder_sharpe` | M1 | 13-arm ladder, both universes, seed error bars |
| `fig_growth_10k` | M6 | growth of GBP 10,000, best baseline vs best ours |
| `fig_regime_variables` | M3 | η² per variable per scheme |
| `fig_regime_ribbons` | part of M2 | weekly labels, dwell and switch rate |
| `fig_regime_agreement` | part of M2 | the three 3×3 co-occurrence tables + AMI |
| `fig_capacity_curve` | M4 | K curve **plus** the val-vs-test scatter, r = −0.477 |

Design decisions worth not re-litigating:

* **Colour is validated, not chosen by eye.** Grey = classical baselines (a
  reference, not a series), orange = our decision-focused arms, blue = learned
  comparisons. The LSTM moved from "ours" to "comparisons" — it is a challenger,
  not one of ours. blue/orange separate by ΔE 24.7 under protanopia against a
  threshold of 8, and differ in lightness so the figures survive greyscale.
* **`fig_growth_10k` uses a min–max envelope** for the other eleven arms rather
  than eleven grey lines, and draws weekly rather than daily: 4,530 points in a
  3-inch panel is finer than print resolves, which is why the old lines looked
  furry.
* **`fig_capacity_curve` is sector11 only.** etf10's 3-seed spreads are
  unreliable (§6.5) and drawing them as wide bands invites reading noise as
  signal.
* **`fig_regime_ribbons` and `fig_regime_agreement` use ordinal ramps**, because
  the regimes within a scheme are ordered (direction, severity, VIX level).
  Labels are semantic — "just fell / drifted / just rose" — not `0/1/2`.
* No figure carries a number in its title. Figure numbers belong in captions.

Caption facts the figures do not state themselves: the ladder's error bars are
**seed dispersion, not significance** (baselines are deterministic, so they show
none, and zero of 24 adjacent comparisons reached significance); the growth
band is the min–max envelope of the other eleven arms.

**One honest oddity to state in the text:** on sector11 the orange line in
`fig_growth_10k` sits *above* buy-and-hold SPY from roughly 2011 to 2021 and
only loses in the final stretch. Now that the lines are legible an examiner will
see it. Frame it as a late-period effect, consistent with the arms being
statistically indistinguishable.

### 11.3 Architecture diagrams — hand-written SVG

`results/analysis_figures/architecture_harness.svg` and
`architecture_model.svg`, both **1600 × 826**, same fonts and stroke weights so
they read as a matched pair.

**Mermaid was abandoned.** Nested subgraph `direction` is overridden whenever
edges cross the boundary, so it could not produce a compact landscape layout.
The SVGs are hand-authored, plain text, and trivially editable: styles are CSS
classes at the top (`.frz` dashed = frozen, `.mdl` heavy = trainable), edges are
`<path d="M x,y H x V y H x"/>` — `H` horizontal, `V` vertical. Zero line
crossings; every label clear of every line.

Conventions the diagrams encode, and the questions they pre-empt:

* Dashed border = frozen or annotation; heavy border = trainable.
* Dashed edge = optional or conditional path; the gradient loop is the thesis.
* Diagram 1 shows the risk-free rate doing **two** jobs — it becomes the cash
  column of the accounting panel *and* is subtracted in the loss. That pairing
  is why an all-cash portfolio scores exactly zero.
* Diagram 1 shows the encoder receiving **rolling z-scored log returns of only
  the assets alive since the universe start** — XLRE and XLC are tradeable but
  never drive the reservoir (R4).
* Diagram 2 carries `∂wₜ/∂w⁽ᵏ⁾ₜ = πₜ,ₖ` as its own box. That identity is the
  specialisation mechanism and is the thing to point at when asked how the model
  knows what a regime is.
* The gate's order of operations was **corrected** against `gating.py:102`:
  noise is added to the **raw** logits and τ_g divides **after**, so the
  effective noise is σ_g/τ_g. At the frozen τ_g = 1.0 it makes no difference;
  at the searched 0.5 it doubles.

⚠️ **`results/meeting_figures/` does not exist.** The five M-series figures live
in `results/analysis_figures/`, but `make_meeting_figures.py:344` still writes
to `meeting_figures/`. §8 step 2 of this file points at the empty path. Running
that script produces a second copy in a second directory.

### 11.4 The demo deck

`~/Desktop/AI/dissertation/deck/` — `build_deck.js` (pptxgenjs, commented,
regenerates in a second with `node build_deck.js`), `assets/`, and
`regime_rc_demo.pptx`. Validation passes; every slide was rendered and checked.

**The edited version is `~/Desktop/dissertation/regime_rc_demo.key`** (Keynote,
8 slides — the Conclusions slide was merged away). The `.pptx` in `deck/` is the
un-edited generated original. Reviewed read-only on 14 Aug; open issues in that
file:

* **The MSE worked example on slide 2 is arithmetically wrong.** Model 2 at
  `A = 7%` gives errors +3 and −2, so MSE = **6.5**, not the 5 shown. More
  importantly the example only bites when both models have the *same* MSE —
  set Model 2's Asset A to **6%** and both come to exactly 4.
* "learned **dates**" for "learned gates" on slide 6.
* Spelling throughout: traning, weather, forcast, furthur, specilization,
  differnet, Statical, benifit, chaning, trands, assett, archtecture, outuput,
  geneunly, creats.
* Page numbers were hand-set and do not follow the removed slide: the last
  slide shows 9.
* Slide 1 reads "for Portfolio Asset Management on 2 universes"; the registered
  title is "for Multi-Objective Portfolio Construction".

Notes on rendering, since LibreOffice is **not installed** on this machine:
export via Keynote (`open -a Keynote`, then AppleScript `export … as PDF`), then
rasterise with `pymupdf`. Two bugs found only by looking at the rendered slides:
architecture PNGs that contained a zoomed corner of the diagram rather than the
whole thing, and images placed with the wrong aspect ratio. `build_deck.js` now
carries an `ASPECT` table and fits every image inside its box.

### 11.5 `portfolio-data` and `portfolio-forecast` — simplified

Both superseded folders were cut down on request. **Nothing in `regime-rc`
depends on either** — `regime-rc/data/raw/` holds a one-time copy of 26 tickers
and there is no code reference to `portfolio-data` anywhere in `src/`,
`scripts/` or `configs/`.

`portfolio-data`: settings are now variables at the top of `main.py` (no
argparse); `logging_utils.py` deleted; `seaborn`, `scipy` and `tenacity` dropped;
the `validate:` config block is optional and skipping it is announced in the log;
`risk_free_rate` is read under either name and is **a percent** (4.0 means 4%).
20 tests pass. Two real bugs fixed: a fresh download and a cache read returned
different index names, and the index checks now raise before the other checks
run (they read rows by date, so on a duplicated index they failed confusingly).

`portfolio-forecast`: 349 lines + a 40-line config became **264 lines in one
file**, settings as variables, `config.yaml` and the empty `src/` deleted.

⚠️ **Setting the risk-free rate to 4% changed that folder's headline.** With
`rf = 0` the ETF strategy appeared to beat buy-and-hold, 0.52 against 0.45. At
4% it is **−0.18 against 0.07**, and the strategy now loses on all three
universes on both risk-adjusted return and money. The old "win" was an artefact
of a zero risk-free rate. `portfolio-forecast/README.md` is updated. If any
number from that folder is cited in the dissertation, cite the new ones.

A top-level `~/Desktop/AI/dissertation/requirements.txt` now pins all 18
packages across the three projects, Python 3.14.3, with the regime-rc pin
winning wherever the sub-projects had drifted.

### 11.6 What P10 did NOT do

* No analysis, no retraining, no new registry rows.
* `results/analysis_figures/M1…M6*.png` are **left in place** for comparison.
* The three §6.9 blockers are untouched: ~69 unregistered trials, the etf10
  capacity curve at 3 seeds, and the still-uncommitted P9 documentation.
* `git status` on `main` still shows `FINDINGS.md` untracked and `HANDOFF.md`,
  `README.md`, `SEARCH_FINDINGS.md`, `results/registry.csv`,
  `src/rrc/eval/registry.py` modified, plus an untracked
  `results/registry.csv.pre-dirty-strip.bak` (1.1 MB — decide: commit as the
  audit trail for the `-dirty` strip, or gitignore).

### 11.7 START HERE — revised for P11

1. **Resolve the three numeric discrepancies in §11.1.** Everything else in the
   results chapter rests on those numbers being consistent between the docs, the
   CSVs and the figures.
2. **Fix the five deck issues in §11.4** — the MSE arithmetic first.
3. **Commit the P9 + P10 work**, then tag `config-freeze`. `results-freeze`
   still waits on §6.9.
4. **Write.** Three findings, two supports, 40 pages. Figures are done — the six
   in §11.2 plus the two diagrams in §11.3 are the full set the report needs.
5. **Holdout last**, ~20–25 Aug, after freezing.

**Still do NOT spend time on** everything in §8's do-not list. Two cheap runs
would genuinely add something if the writing runs ahead, both proposed this
session and neither started:

* **A naive 5-day momentum baseline added to the ladder.** F1 says the gate
  tracks 5-day direction, so the question is whether a one-line rule reproduces
  the best arm. Either answer improves the dissertation: a null strengthens F2,
  a match is a finding.
* **Permute the sentiment column and retrain `K=1 + sentiment`.** It is the best
  learned arm on both universes and the +0.093 that makes it so has never been
  tested against a shuffled control.

And one that is nearly free, and sharpens F1 considerably:
`run_regime_interpretation.py` currently builds `MixtureOfExperts(n_state,
n_assets, N_EXPERTS)` with no `gating_n_inputs`, so **the interpreted gate reads
the reservoir state alone** — it never saw VIX or sentiment. "The gate ignores
risk" is therefore established only for a gate that *could not see* risk. Run
the same analysis on the rung-3a model, whose gate is handed VIX and credit
spread directly. If it still partitions on direction, the claim becomes "it was
given risk and declined to use it", which is much harder to explain away.

---

## 12. P11 — the holdout, F1 closed, and the write-up (session of 2026-08-29)

This session **spent the holdout**, **closed the biggest hole in F1**, added a
theoretical mechanism for it, confirmed the null three more ways, and started the
report. Read §12.1 and §12.9 before touching anything.

### 12.1 ⛔ THE HOLDOUT IS SPENT — it cannot be run again

Run on **2026-08-17** by `scripts/run_holdout.py` (new). Window
**2025-01-02 → 2026-06-30**, 373 trading days, 78 weekly decisions. Train ended
2023-12-21 (embargoed), validation 2024, so nothing about the configuration was
chosen with knowledge of the holdout. Results in `results/runs/holdout/`.

**sector11**

| arm | Sharpe | maxDD | turnover |
|---|---|---|---|
| r0 1/N | **+0.812** | −15.0% | 1.1 |
| **FROZEN K=1 (primary)** | **+0.811** ±0.033 | −13.7% | 3.6 |
| best-found K=2 *(secondary)* | +0.809 ±0.022 | −13.7% | 2.9 |
| r0 buy-and-hold SPY | +0.786 | −18.8% | 0.7 |
| r0 Ledoit–Wolf MV | +0.777 | −14.5% | 5.7 |
| r0 inverse-vol | +0.710 | −14.4% | 1.5 |

**etf10**

| arm | Sharpe | maxDD | turnover |
|---|---|---|---|
| r0 1/N | **+1.340** | −8.9% | 1.0 |
| r0 Ledoit–Wolf MV | +1.313 | −19.5% | 2.0 |
| **FROZEN K=3 (primary)** | **+1.295** ±0.023 | −8.3% | 3.0 |
| r0 inverse-vol | +1.136 | −6.7% | 1.5 |
| best-found K=5 *(secondary)* | +0.930 ±**0.583** | −6.8% | 14.6 |
| r0 buy-and-hold SPY | +0.786 | −18.8% | 0.7 |

**The reading.** The frozen config lands **statistically on top of 1/N** on both
universes (gaps 0.001 and 0.045, a small fraction of the standard error on a
373-day Sharpe). **The holdout confirms the walk-forward null out of sample** —
for a null-result thesis this is the ideal outcome: the untouched test agrees
with the body of evidence instead of contradicting it.

Two supporting points worth writing up:
* **It vindicates the freeze.** The best-found secondary is no better on
  sector11 and clearly worse and wildly unstable on etf10 (±0.583, turnover
  14.6). Putting a searched winner on the holdout would have produced a worse,
  unreliable number. F4 in one line.
* **Small point in the model's favour:** the frozen arm has the shallowest
  drawdown of the high-Sharpe arms on both universes.

**Caveats to state.** The window was **benign** — Sharpes are high everywhere
(1/N at 1.34 on etf10), so "the model does fine" mostly means "everything did
fine"; it is not a stress test. 373 days is one short path with a large standard
error: treat frozen and 1/N as **tied**, not ranked. Internal check passed: SPY
scores identically (0.786) on both universes, as it must.

**Provenance.** `run_holdout.py` contains no registry calls at all, so the run
wrote **no registry rows**, and it made no git commit. The numbers above are nonetheless the real
holdout numbers. If an audit-trail version is ever wanted, it is these exact
figures written with `split="holdout"` — do **not** re-run to "regenerate" them.

### 12.2 ⭐ F1 is closed — the gate was given VIX and still tracks direction

This is §11.7's proposed experiment, executed. `scripts/run_regime_interpretation_signals.py`
(new) trains **both** gates in one run on identical folds/seeds — the state-only
gate and a signal-augmented gate (the 3a routing: gate reads `[state ; signals]`,
experts unchanged). K=3, seeds {0,1,2}, both universes.
Output: `results/runs/regimes_signals/`.

η² (% of each variable explained by the labelling; learned arms are 3-seed means):

| variable | sector11 state → signal | hmm / vix | etf10 state → signal | hmm / vix |
|---|---|---|---|---|
| **return_5d** | **12.5 → 18.4** | 0.2 / 4.5 | **24.8 → 18.4** | 0.1 / 3.2 |
| **realised_vol_21d** | 0.8 → **1.6** | **44.3** / 18.9 | 0.3 → **1.1** | **25.2** / 11.3 |
| short_rate | 0.5 → 8.0 | 2.7 / 0.9 | 0.8 → 9.3 | 1.9 / 0.4 |
| credit_spread | 0.4 → 4.4 | 20.7 / 28.0 | 0.9 → 5.7 | 3.8 / 28.9 |
| vix | 1.4 → 5.9 | 20.7 / 65.4 | 3.5 → 5.0 | 2.5 / 62.8 |

**Handed VIX directly, the gate stays a direction detector.** Direction remains
rank 1 (18.4 on both); realised volatility stays on the floor (1.6 / 1.1, still
~25× below the HMM). AMI to the statistical gates stays ≈ 0
(signal-vs-vix 0.022 / 0.020; signal-vs-hmm 0.007 / 0.006), and dwell stays fast
(1.8→1.9 wks, 1.7→1.8) rather than lengthening toward VIX's 3.3–3.7.

**Why this matters:** it resolves *couldn't* vs *wouldn't*. The claim upgrades
from "the gate tracks direction" (which was near-forced, since it never saw VIX)
to **"it was offered risk and declined it"** — a revealed preference of the
decision objective. It also closes RQ1's "equal information" clause, which the
state-only comparison quietly violated.

**Nuance not to flatten.** The gate *does* read the signals — AMI(state, signal)
is 0.165 / 0.180 and `short_rate` η² jumps most (to 8–9%). But what it takes up
is a **slow macro level**, not risk. **Anchor the claim on `realised_vol_21d`,
not on `vix` η²**: the vix rise (1.4→5.9) is partly mechanical now that vix is an
input, whereas realised vol is computed from returns independently of that
channel and stays at the floor. That is the clean even-function test.

Figure: `results/runs/regimes_signals/M3b_gate_with_signals.png` and
`results/meeting_figures/M3b_gate_with_signals.png`, built by
`scripts/make_signal_gate_figure.py` (new). Four arms in the M3 style; keeps
`short_rate` in the panel (M3 drops it) because that is where the new arm moves.

### 12.3 ⭐ The mechanism behind F1 — an odd/even symmetry argument (theory, new)

Derived this session, not measured, and it is the strongest new material for the
discussion chapter because it is **exact in three lines**.

With `bias_scale = 0.0` and `x_0 = 0`, the reservoir state is an **exactly odd**
functional of the driving returns: flip the sign of every input and every state
flips sign. (Induction: `x_0 = 0 = −0`; assume `x_{t−1} → −x_{t−1}`; then
`W_in(−u) + W(−x) = −(W_in u + W x)` and tanh is odd, so `x_t → −x_t`. The
trailing z-score preserves it — flipping raw returns flips the trailing mean while
the trailing std, being even, does not.)

**Direction is odd; volatility is even.** For a linear gate,
`argmax(W_g x + b_g)` can only be invariant under `x → −x` if the `W_g x` term
never changes the argmax, i.e. the gate is constant. Therefore **a non-degenerate
linear gate on this state cannot produce a volatility partition.**

This predicts exactly the measured profile — the three learned states come out
**up / sideways / down** on 5-day direction (+0.45 / +0.18 / −0.74 at seed 0 on
sector11), which is the only shape an odd state with a trainable logit bias can
make. And it explains the HMM contrast: the HMM's observation vector *contains*
trailing 21-day realised volatility — it was handed a hand-computed even feature
that our gate could never construct.

⚠️ **One existing caveat needs narrowing.** `SEARCH_FINDINGS.md` says "the
reservoir encodes both direction and volatility, so organising on direction is a
CHOICE". A *nonlinear* readout could recover even functions from an odd state;
the **linear** gate we actually use cannot. As written the caveat claims more
than the evidence supports.

**Supporting quantification** (also new, `α = 0.30`, `ρ = 0.90`, pre-tanh std
0.7): linearised, `A = (1−α)I + αW` has eigenvalues `0.7 + 0.27·e^{iθ}`, so
`|λ| ∈ [0.43, 0.97]` and per-mode memory `τ = −1/ln|λ|` spans **1.2–32.8 days** —
reproducing §6.7's own numbers from first principles. The effective tanh gain at
std 0.7 is **0.73**, which shortens `τ_max` to ≈9.2 days, close to the
**measured** 7.7–8.5 day weighted-average gate memory. Energy-coverage reading:
**~90% of the state is the last 3–4 weeks, ~50% the last week**. Two consequences
worth a sentence each: the leak is what *creates* the spread of timescales (at
α=1 every mode has identical τ = 9.5 d), and **α, not ρ, is the binding lever on
memory** (at α=0.3, sweeping ρ 0.5→0.999 moves τ_max only 4.7→11.8 d) — which
explains why the ρ searches kept finding nothing.

### 12.4 VIX in the gate does not help performance either

`scripts/run_vix_on_bestk.py` (new). Each universe's best-capacity config held
fixed, VIX-in-gate added, state-only control re-scored in the same run. 5 seeds,
concatenated walk-forward **test**. Output: `results/runs/vix_bestk/`.

| | gate = state only | gate = state + signals | effect |
|---|---|---|---|
| sector11 K=2 (N_r 300, ρ 0.85) | **0.513** ±0.025 | 0.490 ±0.023 | **−0.023** (paired sd 0.026) |
| etf10 K=5 (N_r 600, ρ 0.95) | **0.408** ±0.106 | 0.201 ±0.070 | **−0.207** (paired sd 0.140) |

**The control reproduces the capacity-curve reference exactly** (0.505 / 0.411 on
the first 3 seeds), which is what makes the deltas trustworthy. Adding VIX is
**neutral to harmful**: nothing on sector11, a real-looking drop on etf10 (though
that baseline is the known-unreliable one). Consistent with §12.2 — a gate that
will not reorganise onto risk cannot profit from a risk signal; the extra inputs
just give a direction-seeking gate more to overfit. Supports F2 and F8.

### 12.5 The fine grid — 1,485 configs, nothing beats the defaults

`scripts/run_fine_grid.py` (new). Dense local grid around each universe's
best-capacity optimum: sector11 810 cells, etf10 675 cells (N_r × ρ dense; α, std,
τ_g 3-point). Validation on the selection folds (2 seeds), then the **top 12
re-scored on the full walk-forward test** (3 seeds). Output:
`results/runs/grid_fine/`; candidate reservoirs go to a scratch dir, **no registry
rows**.

| | best on **validation** | → its test | best **test** in top-12 | capacity ref |
|---|---|---|---|---|
| sector11 | 0.678 (N_r 330, ρ 0.83, α .30, std .80, τ 1.2) | 0.522 | 0.532 | 0.505 |
| etf10 | 0.659 (N_r 600, ρ 0.99, α .20, std .60, τ 1.2) | **0.250** | 0.411 (= the capacity config) | 0.411 |

**Three findings.** (1) **No transferable improvement exists** — sector11's best
test (0.532 vs 0.505) is cherry-picked from 12 test evaluations with sds of
0.02–0.06; etf10's best test *is* the existing config. (2) **Validation still
points the wrong way**, now at high resolution: the val winner transfers to
mid-table on sector11 and *badly* on etf10 (0.659 → 0.250) — selecting off this
grid would give you something **worse than the defaults**. (3) **Finer search only
inflated validation**: the val ceiling rose 0.559 → 0.678 while test stayed put.

**The figure that says it in one picture:**
`results/runs/grid_fine/heatmap_val_vs_test_sector11.png`
(`scripts/make_grid_fine_heatmap.py`, new). The N_r × ρ slice at default
nuisances, val and test side by side. **val argmax (330, 0.92) = 0.636; test
argmax (270, 0.89) = 0.520** — different corners, and the rows nearly swap: the
row that is validation-hot (330) is test-mediocre, the row that is
validation-*coldest* (270) is test-hot. Whole test range 0.39–0.52.

**Write-up value:** this upgrades the freeze from "search didn't help" to *"a
dense grid of ~1,500 configurations found no reservoir that beats the
pre-specified defaults out of sample, and the in-sample winner transferred
worse."* Report it as a **robustness probe of the defaults, not a selection**.

### 12.6 The two loss penalties, re-verified at the frozen config

`scripts/run_penalty_sweep.py` (new). The λ_cvar numbers on record were measured
at P8, *before* the gate-noise anneal fix. Re-checked on etf10 (K=3, where both
penalties act; sector11 is K=1 where λ_balance is inert), frozen reservoir,
5 seeds, concatenated test:

| config | Sharpe | maxDD |
|---|---|---|
| **FROZEN (cvar 0, bal 0.01)** | **+0.237** ±0.025 | −22.9% |
| cvar 0.5 / 1.0 / 5.0 | +0.231 / +0.248 / +0.248 | −22.7 / −22.6 / **−21.2%** |
| bal 0.00 / 0.05 / 0.10 | +0.237 / +0.247 / +0.236 | −22.9% |

**Everything is flat within seed noise** (whole spread 0.231–0.248 < one seed sd).
The write-up can now say **"re-verified at the frozen configuration
post-anneal-fix"** rather than "measured at P8". Two nuances: on etf10 λ_cvar is
*harmless*, not harmful (the "monotonically worse" wording is a sector11
statement); and λ_cvar = 5 trims maxDD 22.9→21.2% at no Sharpe cost — the one
place a penalty earned anything, though not enough to move the frozen choice.

### 12.7 ✅ §11.1 discrepancy #1 is RESOLVED

The docs were right; the old CSVs were seed-0. Verified directly:

| | old `regimes/` (seed 0) | new `regimes_signals/` (3-seed mean) | `FINDINGS.md` quotes |
|---|---|---|---|
| sector11 return_5d η² | 20.9 | **12.5** | **12.5** ✓ |
| etf10 return_5d η² | 29.6 | **24.8** | **24.8** ✓ |

`run_regime_interpretation.py` defaults to `--seed 0`, so its CSVs and every
figure built from them are single-seed, while `FINDINGS.md` quotes the 3-seed
mean. **Nothing is wrong with the finding or the docs** — the two artifacts are
different quantities. Decide one convention and label it; the new
`regimes_signals/` CSVs are 3-seed means and match the docs.
Discrepancies #2 (HMM dwell 12.9 vs 13.1 wks) and #3 (AMI, agrees) are unchanged.

### 12.8 New scripts (all committed)

```
scripts/run_holdout.py                        THE holdout run — do not re-run
scripts/run_regime_interpretation_signals.py  gate WITH signals vs without
scripts/make_signal_gate_figure.py            M3b figure
scripts/run_vix_on_bestk.py                   VIX-in-gate on the best-K configs
scripts/run_fine_grid.py                      the 1,485-cell dense grid
scripts/make_grid_fine_heatmap.py             val-vs-test heatmap
scripts/run_penalty_sweep.py                  penalty re-check at the frozen config
```

New result dirs: `results/runs/{holdout,regimes_signals,vix_bestk,grid_fine}/`.
None of these append registry rows.

**The `Makefile` was rewritten** as a copy-paste cheat sheet: every recipe line is
a literal full command (no variables, no loops), organised in pipeline order
(setup → data & reservoir → ladder → search → evaluation → follow-ups → holdout →
figures). It fixed three previously **broken** targets (`hyper`, `rung5`, `ladder`
called scripts that require `--universe` without passing it) and added the ~25
missing ones. Use `make <target>` or copy any line straight into the terminal.

### 12.9 ⛔ EDITORIAL: do not mention the deflated Sharpe ratio

User decision, 2026-08-29. **The DSR must not appear anywhere in the write-up** —
not in prose, figures, the abstract, or verbal explanation. Drop the "does not
survive deflation" framing and the pre-registered trial-count machinery from the
narrative.

**Argue the null with what remains, which is enough:** the paired stationary
bootstrap and the Ledoit–Wolf robust Sharpe-difference test (every learned arm's
CI straddles zero against the baselines), the held-out verification (six searched
winners, none transferred), the fine grid (§12.5), and the holdout (§12.1).

⚠️ **Scrub still needed.** Drafted passages that currently violate this:
* `report/Preamble/Abstract.tex` — the deflation sentence,
* `report/Chapter1/Chapter1.tex` §Contribution — "nothing survives correction for
  the number of configurations we tried" **and `\cite{bailey2014deflated}`**, and
  RQ2's "survives correction for selection",
* `report/Chapter3/Chapter3.tex` — the inference paragraph in the evaluation
  section,
* `bibliography.bib` still contains `bailey2014deflated` (harmless once uncited).

This also **moots §6.9's trial-count blocker** — with no DSR there is no trial
count to state, so the ~69 unregistered P9 trials no longer gate anything in the
write-up. Keep the registry as a provenance record, not as DSR input.

### 12.10 The report — `~/Desktop/AI/dissertation/report/`

A University of Birmingham LaTeX template, `Dissertation.tex` is the master file.
**No LaTeX toolchain on this machine** (`pdflatex`/`latexmk` absent) — build on
Overleaf or a MacTeX install:

```
cd ~/Desktop/AI/dissertation/report
pdflatex Dissertation && bibtex Dissertation && pdflatex Dissertation && pdflatex Dissertation
```

| piece | state |
|---|---|
| Front matter | title page, abstract, abbreviations, acknowledgements — **done** (user has since rewritten the abstract) |
| Ch1 Introduction | drafted, **user is actively editing** — has the MSE-vs-decision table |
| Ch2 Background | **stub** (section headings only) |
| Ch3 Methodology | drafted, **user has heavily rewritten** — ledger, loss, reservoir, gate, baselines, protocol |
| Ch4 Data & Setup | drafted, **user is actively editing** — universes, signals, availability matrix, hyperparameters, reproducibility |
| Ch5 Results | **stub** — all numbers exist, nothing written |
| Ch6 Discussion / Ch7 Limitations / Ch8 Conclusion | **stubs** |
| `bibliography.bib` | ~14 entries wired to the drafted chapters |

⚠️ **The user edits these files directly.** Always re-read a chapter before
touching it; do not regenerate one from an earlier draft.

Figures in `report/figures/`: `fig_architecture.png`, `fig_walkforward.png`
(protocol timeline), `fig_data_coverage.png` (asset availability + firewall),
`fig_signals.png` (the five z-scored signals). The user also has `arch1.png`
there, which Ch3 references.

Two citations verified against the publisher this session, for the regime claim:
**Guidolin & Timmermann (2007)**, *JEDC* **31**(11), 3503–3544 (free copy at the
St. Louis Fed), and **Ang & Bekaert (2002)**, *RFS* **15**(4), 1137–1187. Neither
is in `bibliography.bib` yet.

### 12.11 Demo / viva material produced

A 10-minute demo script written to the deck's slide order (title → motivation →
pipeline → model detail → growth results → η² → capacity/transfer → close), plus
an anticipated-Q&A set. The deck is `~/Desktop/dissertation/regime_rc_demo.key`.
Factual corrections made to the user's spoken draft that are worth remembering:
the ledger is **PyTorch, not NumPy** (you cannot backprop through NumPy); Dow-30
exclusion is **survivorship** bias; sentiment is **one of** the five signals; and
the SCR 2024 result is *universal approximation of fading-memory filters*.

### 12.12 START HERE — the only remaining work is writing

Today is **Saturday 2026-08-29**. The report is due **Tuesday 1 September**.
**The analysis is finished. Do not start new experiments.**

1. **Scrub the DSR references** listed in §12.9. Small, mechanical, and the
   passages are already drafted.
2. **Write Chapter 5 (Results).** Every number exists: the walk-forward ladder
   and growth figures, η² (§12.2 and `results/runs/regimes/`), the capacity curve
   and its val-vs-test anti-correlation, and the holdout (§12.1). Lead with F1
   (now closed by §12.2), then F2, then F4.
3. **Write Chapters 6–8** (Discussion, Limitations, Conclusion). §12.3's symmetry
   argument is the discussion chapter's strongest paragraph; §12.5 belongs in the
   transfer-failure section; the limitations list is in §7 of this file, plus the
   benign-holdout caveat from §12.1.
4. **Chapter 2 (Background)** last — it is the most mechanical.

**Do NOT re-run or re-litigate:** the holdout (spent), more loss terms, NSGA in
any form, seed ensembling, forcing expert specialisation, finer hyperparameter
grids (§12.5 closed this), or VIX-into-gate variants (§12.2 and §12.4 closed
those). Everything in §8's and §11.7's do-not lists still stands.

**If the writing somehow runs ahead**, the two genuinely open items are the
sentiment-permutation control on `K=1 + sentiment` (§11.7) and the etf10 capacity
curve at 5+ seeds — but the simpler course is to **not quote etf10's per-K
numbers** and say its 3-seed spreads are unreliable, which removes the last
must-run item entirely.
