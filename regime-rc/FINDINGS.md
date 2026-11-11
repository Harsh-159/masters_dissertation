# FINDINGS — every result, decision, reversal, bug and surprise

_Assembled 2026-08-07, covering P0 (13 July) to P9 (7 August);_
_**section 10 appended 2026-08-29 (P11)**._

> **Read §10 first.** It closes F1 (the gate was given VIX and still tracked
> direction), adds the symmetry mechanism behind it, records **the holdout, which
> is now spent**, and carries a standing editorial rule: **do not mention the
> deflated Sharpe ratio in the write-up** — which retires L4 and L5 below from the
> narrative.

This is the **research record**, organised by *kind of finding* rather than by
session, and it exists because the material was previously scattered across four
places: `SEARCH_FINDINGS.md` (searches only), `HANDOFF.md` (organised by
session, mixed with logistics), ~14 module docstrings (invisible unless opened),
and commit messages.

It invents nothing. Every entry points at where the evidence lives.

**How to use it.** Sections 1–3 are the dissertation's *results*. Sections 4–7
are the *methods chapter* — the decisions, reversals, bugs and lessons that
explain why the results should be believed. Section 8 is what remains open.

**Notation.** ⭐ = headline material. 🔁 = a position we reversed on evidence.
🐛 = a bug caught, with what it would have cost. ❌ = something tried and
rejected. ⚠️ = a caveat that must be stated in the write-up.

---

## 1. The research questions, and what we can now say

### ⭐ F1 — Decision-defined regimes are orthogonal to statistical ones, because they measure a different quantity

**RQ1 asked** whether regimes discovered by the decision objective carve the
market differently from statistically-defined ones. **They do, and we can now
say why.**

> ✅ **STRENGTHENED AT P11 — see §10.1 and §10.2.** The gate analysed below reads
> the reservoir state *only*, so "it ignores risk" was established for a gate that
> could not see risk. It has since been re-run with VIX and the four macro signals
> fed directly to the gate: **it still organises on direction**. The claim is now
> "offered risk and declined it", and §10.2 derives *why* from the architecture.

| evidence | number | source |
|---|---|---|
| the observation | AMI ≈ 0 against HMM and VIX, every fold, both universes | P7, `eval/agreement.py` |
| **the mechanism** | statistical gates track **risk** (volatility 44.3%, dispersion 36.2%, VIX 65.4%); the learned gate tracks **direction** (5-day return 12.5% sector11 / 24.8% etf10) | `run_regime_interpretation.py`, F9/M3 |
| each blind to the other | HMM explains 0.1–0.2% of recent direction; the learned gate 0.3–0.8% of volatility | same |
| the consequence | dwell follows the variable: learned 1.8 wks (direction flips), HMM 12.9 wks (volatility persists), VIX 3.7 wks | `dwell_and_transition` |
| the control | the two **risk** detectors agree with *each other* (AMI +0.093 sector11) ~20× more than either agrees with the learned gate (≈0.00) | M2 |
| robustness | direction ranks **first** for the learned gate in all 6 runs (2 universes × 3 seeds); runner-up never exceeds 5.2% | `separation_*.csv` |

**What the regimes actually mean.** Learned: *"the market just rose / drifted /
just fell"* (+0.57σ, −0.24σ, −0.51σ on 5-day return, everything else < 0.3σ).
HMM: a severity ladder *calm / stressed / crisis* (volatility −0.40σ, +0.96σ,
+1.62σ) **with direction flat at ±0.03σ** — it literally cannot tell you which
way the market went.

**Why the control panel matters.** Without it, AMI ≈ 0 is ambiguous — maybe AMI
is always ≈0 for any two 3-state labellings of noisy financial data. The
HMM-vs-VIX panel is a **positive control**: two schemes built completely
differently (one a learned Gaussian mixture, one a fixed rule on an external
index) that nonetheless agree. The instrument detects agreement when agreement
exists. Against that, ours registers nothing.

⚠️ **Two caveats that must be stated.** (a) This runs at K = 3, but sector11's
frozen configuration is K = 1, where the gate is mathematically inert and there
are **no regimes to interpret**. (b) The "statistical schemes cluster together"
sub-claim holds on sector11 (0.093 vs 0.004/0.008) but **not** on etf10 (0.017
vs −0.002/0.020). The orthogonality holds on both; the clustering does not.

### ⭐ F2 — The carving does not pay, on either universe, under any test

| | sector11 (18 folds, 2007–24) | etf10 (10 folds, 2015–24) |
|---|---|---|
| best baseline | inverse-vol **0.511** | buy-and-hold SPY **0.688** |
| best learned arm | ours K=1 + sentiment 0.507 | ours K=1 + sentiment 0.318 |
| gap | −0.004 (a tie, at 10× the turnover) | **−0.370 (a rout)** |
| significant adjacent-rung differences | **0 of 12** | **0 of 12** |
| arms clearing deflation | **none** | **none** |
| £10,000 → | SPY £54,551 vs ours £45,998 | SPY £34,059 vs ours £15,545 |

Full ladder, test, net of 5 bp: see `results/runs/walkforward/summary_*.csv`.

**The reframe that keeps etf10 honest.** 1/N scores 0.374 there and inverse-vol
0.278. **Every** diversified portfolio lost to SPY over 2015–2024 — bonds fell
as rates rose, commodities went nowhere, EEM and VGK lagged. The models sit next
to 1/N because that is what they learned to be. Not uniquely broken.

**The 2008 story (sector11).** Every arm sits near +0.43 after 2007, collapses to
about −0.50 after 2008, and takes until **2013 — five years —** to climb back
above +0.35. Without 2008, sector11 validates around +0.83; with it, +0.26. Only
the cumulative-Sharpe view (F4/M-series) shows this; a per-year table cannot.

### F3 — Sentiment is a *routing* signal, not an allocation signal

Rung 3a (signals → gate) is the best sentiment arm on both universes' validation
means and the only one that ever beat plain rung 2 (sector11 F02: 1.48 vs 1.22).
3b (signals → experts) and 3c (both) help nowhere.

⚠️ **But selecting K = 1 forecloses the hypothesis.** At K = 1 the gate is a
softmax over one logit — always 1.0 — so sentiment-into-the-gate is a
**mathematical no-op**. The only route sentiment can take into a K = 1 system is
through the expert. This is a consequence of the K result, not a design choice,
and it must be said plainly.

---

## 2. The methodological finding

### ⭐ F4 — Hyperparameter selection does not transfer on this problem, for either encoder

Six configurations were selected by maximising validation Sharpe, then re-scored
on folds excluded from every selection step (sector11 F13–F18, etf10 F06–F10).

| # | universe | encoder | on validation | on held-out folds |
|---|---|---|---|---|
| 1 | sector11 | reservoir | 0.817 | 0.298 across seeds |
| 2 | sector11 | reservoir | won at K = 3 | died on the move to K = 1 |
| 3 | sector11 | reservoir | 0.603 | 0.526 — **worst of three** |
| 4 | sector11 | reservoir | 0.582 | 0.428 — **worst of three** |
| 5 | etf10 | reservoir | 0.464 | 0.783 vs defaults' 0.809 |
| 6 | both | **LSTM** | 1.011 / 1.068 | 0.738 / 0.473 |

**The decisive measurement:** out of sample, the spread **across** configurations
(0.064) is **smaller** than the spread between two **seeds** of one configuration
(0.085). Which settings you pick matters less than which seed you draw.

**The sharpest instance — the capacity curve.** On sector11, validation and test
Sharpe **anti-correlate across K: r = −0.477**. K = 4 has the best validation of
all eight capacities and mid-table test; K = 2 has near-worst validation and the
**best** test. Selection does not merely fail to help — **it points the wrong
way.** (It does *not* replicate on etf10, r = +0.678, so this is a sector11
finding, not a law.)

**Consequence:** the frozen configuration is the plan's Appendix B **defaults**
on both universes — the only choice requiring no look at any data, and therefore
the only one carrying no selection bias.

⚠️ Nuance not to flatten: on sector11 the winners came out **worst** (evidence of
failure). On etf10 the searched config came out +0.118 *ahead* but inside a
0.150 seed spread — that is **"no evidence it transfers"**, not "evidence it
fails".

---

## 3. Architecture and capacity results

### 🔁 F5 — K = 1 was selected, then the selection was overturned

**What P8 found.** K = 1 won all three rounds of the K search on sector11
(0.611 / 0.580 / 0.670 against K = 3's 0.541 / 0.545 / 0.639), and train Sharpe
did *not* improve with K — the extra experts bought no fitting power, only
variance. Three checks said it was not a training defect: the mechanism works on
etf10 where experts do specialise; forcing specialisation with `lambda_div` made
it monotonically worse; removing gate noise changed it by 0.001.

**What P9 found.** That comparison ran every K at **one** reservoir, tuned for
one of them. Give each K its own reservoir search (9 cells: N_r × ρ) and score
on the full walk-forward test split:

| sector11 test | K=1 | K=2 | K=3 | K=4 | K=5 | K=6 | K=7 | K=8 |
|---|---|---|---|---|---|---|---|---|
| Sharpe | **0.403** ← worst | **0.505** ← best | 0.474 | 0.485 | 0.459 | 0.439 | 0.455 | 0.454 |

**K = 1 is the worst capacity on test.** Resolved beyond two seed spreads
against it: K = 2 (+0.102), K = 5, K = 7, K = 8.

**K and the reservoir genuinely interact** — the searched width climbs with K
(300, 300, 150, 600, 300, 600, 600, 600). Any K comparison at a fixed reservoir
was measuring the wrong thing.

⚠️ **The frozen config nevertheless stays at K = 1.** Re-selecting on a test-set
curve is exactly the bias six held-out checks were run to avoid. Report the curve
as a finding; do not act on it, and say so.

⚠️ etf10's seed spreads reach 0.155 at 3 seeds and are unreliable — a re-check of
K = 7 at 8 seeds moved it 0.215 → 0.276 and sd 0.007 → 0.088. **Re-run at 5+
seeds before quoting.**

### F6 — Experts specialise where structure exists, and it still does not pay

The G2 split verdict, replicated independently at P8:

| universe | max pairwise expert correlation (K=3) | gate entropy | verdict |
|---|---|---|---|
| etf10 | **0.379** (0.169 at K=2) | 0.973 | experts genuinely differentiate |
| sector11 | **0.691** (0.63–0.84) | 0.805 | experts are near-clones |

**The mechanism works where the universe supports it.** But on etf10 — where they
*do* specialise — K = 3 is the **worst learned arm** (0.237). Specialisation
happens and does not convert into out-of-sample value. That is a cleaner
statement of the mixture null than sector11 gives, because there you cannot tell
whether the mechanism or the universe is at fault.

### F7 — The frozen encoder is not beaten by a trained one (D2), now tested fairly

**The fairness problem.** The ladder had **238 reservoir-side search trials
against 0 for the LSTM** — one hand-picked configuration. "Frozen beats trained"
was not a claim that survived that.

**After a matched search** (hidden × lookback, then layers/dropout/lr):

| effect, averaged | sector11 | etf10 | vs median seed sd |
|---|---|---|---|
| **capacity** (hidden 16→128) | **+0.456** | **+0.347** | 0.036 / 0.090 |
| **memory horizon** (lookback 20→250) | +0.067 | +0.022 | — |
| ratio | **6.8×** | **15.5×** | |

**Two findings.** (a) **Capacity matters, and the original hidden = 32 was a real
handicap** — which *contradicts* the argument that fixed it ("a fold holds only a
few hundred weekly decisions; a large network would lose for the wrong reason").
(b) **The memory horizon barely matters**, and shorter is if anything better.

**This answers "did you cripple the LSTM?" with a number.** Corroborated
independently: the reservoir's measured memory capacity at operating settings is
**19.53, with 69% of it inside 60 days** — so the 60-day window was never the
constraint. Neither champion transferred, so D2 survives.

⚠️ State as a limitation: on the folds a practitioner would have searched,
hidden = 128 beats the ladder's hidden = 32 by ~0.5, so the LSTM arm in the
ladder runs at a configuration the search calls poor.

### F8 — Capacity, not signal, is the binding constraint — and the gap is non-stationarity

Five loss regularisers were implemented, measured on the concatenated metric, and
**all five select to zero**:

| term | 0 | small | large |
|---|---|---|---|
| `lambda_cvar` | **0.526** | 0.505 | 0.449 |
| `lambda_turn` | **0.526** | 0.403 | 0.412 (turnover 9.2 → 1.8) |
| `lambda_div` | **0.547** | 0.526 | 0.443 |
| `weight_decay` | **best at 0** | — | — |
| `lambda_drawdown` | **best at 0** | worse Sharpe *and* no drawdown reduction | |

**This is one coherent finding, not five disappointments.** The train/validation
gap is **not fixable overfitting — it is non-stationarity.** Patterns fitted on
the training fold genuinely do not transfer to the next year, so no penalty
computed on the training window can help.

**`weight_decay` is the cleanest proof:** it does exactly what capacity control
should — closes the gap from **0.84 to −0.11** — and validation still gets
*worse*.

Supporting: params-per-decision moves monotonically with K (6.5 → 19.6 → 39.2 at
K = 1/3/6) and so does the gap (−0.006 → +0.021 → +0.047), while **train Sharpe
does not improve** (0.482 → 0.476 → 0.462).

### F9 — The gate does **not** exploit the reservoir's frequency decomposition (a failed prediction)

**Registered before running:** the cycle matrix stays circulant under leaking
(A = (1−α)I + αρP), so ring modes are exactly the Fourier basis with known
memories τ_k spanning **1.2 to 32.8 days**. Since the gate tracks 5-day
direction, it *should* concentrate on short-memory modes.

**It does not.** The trained gate's spectrum is indistinguishable from random
initialisation — weighted-average memory 7.7 → 8.5 days (sector11), 8.1 → 8.4
(etf10). On etf10 the largest per-mode shift (0.0032) is **smaller than a flat
spectrum's own bin height** (0.0066).

**What it constrains.** The gate tracks direction but not by *timescale
selection*. The slow-weighted effective read (~12 days) comes from the **state**
spectrum (42–43% slow) while the readout reads uniformly — the reservoir's
dynamics set the timescale, the readout accepts it.

⚠️ Exact only for the linear system; this reservoir has tanh. Input scale is
calibrated to pre-activation std 0.7 (mildly curved, not saturated), so a
first-order reading is defensible, and the empirical state spectrum is included
so the conclusion does not rest on the linearisation alone.

---

## 4. Design decisions, and the reason for each

| # | decision | why, and the alternative rejected |
|---|---|---|
| D1 | **Frozen reservoir, trainable heads only** | Classical RC recipe; trains in seconds; when something breaks the diagnostics are interpretable. Depth added only if linear demonstrably saturates. |
| D2 | **Simple Cycle Reservoir over a random ESN** | Ring topology makes the spectral radius **exact by construction** — every eigenvalue is ρ times an N-th root of unity, so no rescaling step, no eigenvalue solve, no numerical drift. Random ESN retained as a robustness check (SCR↔ESN canonical correlation 0.987–0.989). |
| D3 | **Cache reservoir states** | Not an optimisation — it is what makes the schedule possible. ~1,000 training runs × 7,000 days of reservoir driving would be minutes each; with cached states a run is seconds. |
| — | **Softmax output layer** | Feasibility baked in: long-only and fully invested **by construction**. No projection step, no constraint solver, smooth gradients. |
| — | **Ledger written in torch** | The ledger *is* the forward pass of the loss. Gradients must flow from every daily return back to the head, so autograd does it for free. |
| — | **Ledger is model-ignorant** | It consumes a weight schedule and knows nothing about reservoirs or gating. That is what lets the trained system and every baseline share **one** accounting implementation, so comparisons can never diverge on bookkeeping. |
| — | **Closed-form holding-period ledger** | Inside a holding period the daily ledger is closed-form: one cumulative product and one dot product replace a day-by-day Python loop. |
| — | **Two ledger implementations** | A readable reference loop and a vectorised fast path, tested equal to **1e-12**. The readable one is the specification. |
| A3 | **Full-span batching, never mini-batches** | The Sharpe objective couples every day through its denominator; a batch smaller than the fold silently changes what is being optimised. |
| A9 | **Cost = c·Σ\|Δw\| (full notional); turnover = ½Σ\|Δw\|** | Three conventions coexist in the literature and mixing them corrupts comparisons invisibly. The factor-of-two difference is deliberate — using the turnover number to compute costs would halve them silently. |
| E7 | **Signal at Friday close → execute Monday close → earn from Tuesday** | The lag lives in `weekly_execution_schedule`, **outside** the ledger, so it cannot be silently changed by a model. Test C7 protects it. |
| A4 | **r_f from DGS3MO, not DTB3** | DTB3 is quoted on a **discount basis**, not as a yield. Every excess return is measured against this series, so a systematic bias in it biases every Sharpe, every loss, and the cash column. Conversion is compounding-consistent: `(1+y/100)^(1/252) − 1`, not `y/100/252`. |
| E1 | **Cash excluded from the MV covariance** | Cash has zero variance, which makes the covariance singular. Instead: optimise over risky assets with `sum(w) ≤ 1` and let the residual sit in cash. Because returns are measured in excess of that same r_f, cash becomes the zero-excess, zero-risk asset automatically. |
| R4 | **Late entrants tradeable but never drive the reservoir** | The reservoir needs a fixed-width driving series across 1999–2026. If the input dimension changed in Oct 2015, every cached state before and after would be incomparable. Output space stays the full universe. |
| — | **Calendar = union of days at least one asset traded** | `pd.bdate_range` *invents* trading days (holidays differ by year and country), and filling invented rows is indistinguishable from fabricating data. Observed, not assumed. |
| — | **Cache each ticker's ENTIRE history** | Changing the study window never triggers a re-download, so folds and the holdout can be re-cut freely; and the pipeline runs offline after the first pull (yfinance rate-limits and its schema drifts). |
| — | **Log returns for the reservoir, simple returns for accounting** | A portfolio's return is the weighted sum of its assets' *simple* returns; log returns are additive over time and better behaved as reservoir input. Mixing them is a subtle, very real error, so they are produced separately and named unambiguously. |
| — | **Per-segment fresh start** | Every evaluated year starts in cash, for models and baselines alike — no continuity advantage anywhere. |
| — | **Unusable reservoir rows stored as NaN, not 0** | A zero flows silently through a matrix multiply and produces a plausible number; a NaN makes accidental use immediately obvious. |
| — | **Availability mask is additive −1e9 on logits** | `exp(−1e9)` underflows to **exactly** 0.0, not to something merely small. Same constant used in the data layer, so there is one definition of "unavailable". |
| R6 | **Equal conditions by literal code reuse** | All six arms go through one training loop. Not two implementations believed to match — the same code path. |
| — | **Registry has no update or delete** | A wrong result is corrected by *appending* a row with an explanatory note, never by editing history. |
| — | **DSR trial count pre-registered before any experiment** | Distinct `cfg_hash` with `split == "val"`. Writing the rule down first is what makes the deflation honest rather than theatre. |
| — | **Both universes never aggregated** | Different calendars (etf10 from 2006, sector11 from 1999); pooling averages 2008 with 2015. |
| — | **Seeds {0..4}, complete seeding of python/numpy/torch** | Error bars are fiction if any RNG is left unseeded; "seed 3" must name the same experiment every time. |

---

## 5. Reversals — positions changed on evidence

### 🔁 R1 — The metric was wrong, and it was material

**Was:** mean of per-fold Sharpes. **Now:** one Sharpe on the concatenated
stream.

Averaging ratios across periods with different denominators is not a valid
aggregation. It discards between-year variation (law of total variance) and
over-weights calm years.

| | mean-of-folds | concatenated | difference |
|---|---|---|---|
| sector11 F01–F03 | +0.429 | −0.053 | **0.48** |
| etf10 `moe_k3` test | +0.481 | +0.241 | 0.24 |
| etf10 `single_head` test | **−0.404** | **+0.310** | **sign flip** |

⭐ **The sign flip is the best single illustration for the methods chapter: an
invalid aggregation does not merely shift a number, it can reverse a
conclusion.** Corroborated by Sharpe's own definition (a single series) and by
arXiv:1802.04413, which flags averaging as incorrect because the Sharpe is a
non-linear statistic.

### 🔁 R2 — NSGA-II built in full, measured, and deleted

Not simplified away — **built, measured, retired**, and `pymoo` dropped.

| schedule, identical folds/model/ledger | etf10 | sector11 |
|---|---|---|
| A: mixture blend | +0.856 | +0.332 |
| B: blend + cardinality only | **+0.909** | **+0.342** |
| C: NSGA front | **−0.325** | **−0.231** |

**Cardinality is free; the front's optimisation destroys ~1.2 Sharpe.** The cause
is conceptual: the front re-picks assets from an EWMA momentum estimate,
discarding the trained system's own view. **It bolts a predict-then-optimise step
onto the end of a decision-focused system** — precisely the pattern the thesis
argues against — and it costs what the thesis predicts. That is a *result*.

Also: choosing 5 of 10 assets is 252 combinations, convex once the subset is
fixed. A stochastic population search was the wrong tool for a problem that
small.

### 🔁 R3 — C2 redefined as the conditioning edge

The plan's literal shuffle-collapse test **failed on etf10** — but for the wrong
reason: shuffling preserves drift, so a static strategy keeps its Sharpe and
buy-and-hold would be "flagged" as leaking.

**The correct quantity is the conditioning edge** = dynamic Sharpe − the
strategy's own static counterpart. Tests prove power both ways: a *planted* leak
survives shuffling in every world and is caught; a zero-weight static head is
**not** accused.

Result: etf10 edge −0.004 (the rung-1 head is essentially a static defensive
tilt — the honest null), sector11 +0.032.

### 🔁 R4 — "The bottleneck is the absence of a defensive action" — refuted

**Was (P8 §5.9):** sector11 is all-equity, cash is the only defensive asset, and
across 1999–2007 holding cash mostly lost, so no expert learns to want it. The
2008 diagnosis was detailed and convincing: mean cash held through 2008 was
**0.095**, flat through October, and the mixture rode the crash to −29.8%.

**Refuted at P9:** etf10 *has* TLT, IEF and GLD — and the models lose by a
**wider** margin there (0.318 vs SPY's 0.688, roughly half the money). Giving the
system a defensive action does not rescue it. **The limitation is not the
universe.**

⚠️ Killing our own hypothesis with evidence is worth reporting. Recycling it
after it failed is not.

### 🔁 R5 — Rung 5 redefined from NSGA front to exact cardinality

Keep the trained blend's top-`K_card` names, renormalise, same ledger. No
retraining, no optimiser. **The measured price of a tradeable book:**

| K_card | 3 | 5 | 7 | 9 | unconstrained |
|---|---|---|---|---|---|
| Sharpe | 0.546 | 0.588 | 0.637 | 0.663 | **0.670** |

Cardinality costs ~0.08 Sharpe at K_card = 5, rising to 0.12 at 3, monotone in
tightness. That quantified price is what rung 5 exists to measure.

### 🔁 R6 — The `-dirty` SHA marker removed

Every development run is dirty — you write, run, look, tweak, run again — so the
marker applied to **2,960 of 2,960 rows** and distinguished nothing. Removed from
`git_sha()` and stripped from the registry; the weaker guarantee (the SHA pins
the *base commit*, `cfg_hash` pins the configuration exactly) is now stated
plainly in the README instead.

---

## 6. Bugs caught, and what each would have cost

| 🐛 | bug | what it would have cost |
|---|---|---|
| **P2** | **All-cash portfolio reported Sharpe 4.17.** Excess returns are zero only in exact arithmetic; in float64 they are ~1e-19 round-off, and the ratio of two such numbers is an arbitrary O(1) value. | A "strategy" that holds cash forever would have looked like the best result in the dissertation. Fixed with an ε = 1e-6 denominator, the **same** ε in the differentiable version — so the quantity reported is the quantity optimised. |
| **P3** | **Memory capacity measured INVERTED.** Probed at the operating input scale; MC measures *linear* recoverability, and driving the reservoir hard puts it in the nonlinear regime where it trades memory for computational richness. | The reservoir would have looked broken, and the response — change the reservoir — would have been exactly wrong. Fixed with probe scale 0.05; F2 now rises 9.5 → 19.5 peaking at ρ = 0.9. A regression test pins the probe-scale dependence. |
| **P7** | **`hmmlearn` returns (K,d,d) full matrices even with `covariance_type="diag"`.** | Silently wrong HMM state probabilities in the single comparison the thesis rests on. Fixed with `diagonal_variances()`; never reshape `covars_`. |
| **P1** | **`pd.NA` broke float casting on exactly the sentiment-outage path.** | A crash, but only on the rare path — so it could have survived into a long run. Fixed to `np.nan`. |
| **P0** | **Unanchored `data/` in `.gitignore` silently untracked `src/rrc/data/`.** | An entire source package missing from version control. Fixed to `/data/`. |
| **P8** | **NSGA front's scale degeneracy.** `mu^T w − gamma·CVaR` compared a *daily* EWMA mean (~2 bp) against a *daily* CVaR (~1.3%), so every risky point scored negative and cash scored exactly 0. | The front locked into all-cash on date 1 and never left — turnover exactly 0.000, Sharpe 0.000, on every fold and seed. Would have been reported as "the front prefers cash". |
| **P8→P9** | **R2's gate-noise anneal never completed.** Targeted `max_epochs/2` (epoch 250) while early stopping fired at epochs 7–80. | **Every mixture ever trained ran at ~90% of full gate noise for its entire life** and never committed to a routing. Affects every rung-2/3/4 number in the registry. Fixed at P9 to `2 × patience`, with two regression tests. P8 and P9 numbers are therefore not strictly comparable at those rungs. |
| **P9** | **Ledoit–Wolf test returned p = 0 comparing a stream against itself.** The gradient is antisymmetric across duplicated moment pairs, so the HAC variance collapses to exactly zero — 0/0 fell through to `inf`. | "Infinitely significant" for a null difference. Caught by a test written *before* it was known to fail. Now returns p = 1. |
| **P9** | **Lossy stream filenames broke a table join silently.** Path-safe names strip `(`, `)`, `/`, so `ours K=1 (rung 1)` → `ours_K=1_rung_1` and the money table could not match the summary table. | A results table with silent NaNs. Fixed with an explicit `arms.json` manifest. |
| **P9** | **`OPTIMAL_K = 1` hardcoded for both universes**, so etf10's arm was labelled "ours-optimal" using a **sector11** selection that no etf10 evidence supports. | A headline table claiming a configuration was optimal on a universe where nothing selected it. Renamed to neutral `K_SINGLE`/`K_MIXTURE`. |

---

## 7. Methodological lessons

### ⭐ L1 — Small samples do not add noise; they manufacture findings

**Three independent instances in this project:**

1. **The first random search** scored 50 candidates at **one seed** and froze the
   maximum at validation 0.817. Re-run on five seeds it scored **0.298**, and
   with seed 0 excluded it was indistinguishable from the default (0.168 vs
   0.165). The entire advantage was one seed on one fold.
2. **The Level-2 frequency smoke test** on 2 folds showed the trained gate
   fleeing the fast modes (25.9% vs the untrained 45.6%) — a clean,
   publishable-looking effect, in the **opposite** direction to the prediction.
   On all 18 folds it evaporated to 42.6% vs 45.6%.
3. **A 3-seed standard deviation is nearly worthless.** etf10 K = 7 measured
   sd 0.007 at 3 seeds and **0.088 at 8** — twelve times larger — and the point
   itself moved 0.215 → 0.276. A 3-sample sd has ~50% coefficient of variation.

⭐ **The pre-registered DSR discipline caught the first one before it reached the
results.** That is the methods-chapter story.

### L2 — Validation is a number that was *maximised*; test is one that was *measured*

Anything selected as the best of many is optimistically biased. In this project
validation is selected **twice over**: early stopping keeps the best of ~500
epochs, and the inner search keeps the best of 9 configurations. The gap between
validation and test **is** the overfitting — and on sector11 the two
**anti-correlate** across K, so the bias is not merely optimistic but
directionally wrong.

### L3 — Walk-forward folds overlap by construction, and it must be quantified

Year Y is fold *i*'s validation and fold *i−1*'s test, so **7 of 18 sector11 and
4 of 10 etf10 test years were visible to hyperparameter selection.** Model
parameters are unaffected (each fold trains strictly on prior data), but
selection saw those years.

**Measured, restricted to years no search ever saw:** learned arms shift +0.02 to
+0.06; **baselines shift more** (SPY +0.21), so the gap to buy-and-hold *widens*.
**The overlap does not flatter us.** Volunteering this reads as rigour.

### L4 — The deflation's trial count is pre-registered; its dispersion is not

> ⛔ **RETIRED FROM THE WRITE-UP (P11, §10.6).** The deflated Sharpe ratio is
> not mentioned in the dissertation. Kept here as a record of what was computed.


The rule fixes **N** (distinct `cfg_hash`, `split == "val"`) and says **nothing
about σ** — yet σ dominates, because the benchmark is proportional to it and only
*logarithmic* in N (50 → 305 trials raises the bar only 27%; σ 0.117 → 0.365
raises it 211%).

Two estimators, both reported:

| estimator | σ | benchmark |
|---|---|---|
| pooled (everything) | 0.365 | **1.058** |
| coherent, sector11 | 0.206 | 0.598 |
| coherent, etf10 | 0.117 | 0.340 |

The coherent estimator's three exclusions are each justified by a convention
locked **before** this calculation existed. **Report the pooled number first**,
then argue for the coherent one — a reader who computes the sceptical number and
finds it already in your table will trust the rest.

### L5 — Baselines must not be deflated

> ⛔ **RETIRED FROM THE WRITE-UP (P11, §10.6).** The deflated Sharpe ratio is
> not mentioned in the dissertation. Kept here as a record of what was computed.


The DSR corrects a result for *the search that produced it*. Nothing selected
buy-and-hold SPY from 305 candidates — it was written into the plan at P4. Its
selection bias is zero, so it is scored at zero dispersion, which reduces the DSR
to the probabilistic Sharpe ratio. Deflating it would be a category error.

### L6 — Chance-corrected metrics are load-bearing

Raw mutual information is **positive even for two random labellings**, and the
bias grows with cluster count (2 clusters → MI 0.0005; 10 → 0.0440). AMI
subtracts the expected value under random labellings with the same cluster sizes,
so the floor sits at exactly 0 regardless. Without it, "AMI ≈ 0.004" could not be
distinguished from the chance floor — and this project's HMM sometimes
degenerates to 2 states, so different folds would have different floors.

### L7 — √252 assumes independence, and inference should not

Measured first-order autocorrelation is ≈ **−0.10**, and the variance ratio at
h = 252 is **0.29–0.86** (below 1), so √252 is mildly **conservative** here — the
bias runs in our favour. We keep the convention for comparability, and inference
uses **HAC-robust** standard errors that make no independence assumption.

⚠️ The h = 252 variance-ratio estimate rests on only ~17 independent year-blocks
and is far too noisy to apply as a correction. The *direction* is solid; the
magnitude is not.

### L8 — The instrument needs a positive control

"Our gate agrees with nothing" is ambiguous until you show the measurement *can*
detect agreement. The HMM-vs-VIX pairing is that control: two schemes built
completely differently that nonetheless agree (AMI +0.093 on sector11).

---

## 8. Open items and honest caveats

| item | status |
|---|---|
| **~69 P9 trials unregistered** | The verification, joint and LSTM searches ran with `--dry-run`. The registry reads **449**; the true count is ~**518**. Register them or state it. **Do not quote 449 as complete.** |
| **Capacity curve at 3 seeds** | etf10 spreads unreliable (K = 7 artefact). Re-run at 5+ seeds before it goes in the report. |
| **Headline results have no registry rows** | `run_walkforward.py`, `run_regime_interpretation.py` and `run_inference.py` write none. The registry's rung-0/1/2/3/4 rows are the **P4–P7** runs at old configs, pre-metric-correction and pre-anneal-fix. |
| **Single-path purged walk-forward, not CPCV** | No distribution of backtest paths, no PBO statistic. Standard practice, but name the gap. |
| **LSTM is stateless across windows** | Hidden state resets each 60-day window; a strong baseline would carry it with truncated BPTT. |
| **Level-1 interpretability runs at K = 3** | The frozen sector11 config is K = 1, where the gate is inert. |
| **Stage-2 LSTM search is one-at-a-time** | Assumes layers/dropout/lr do not interact strongly with capacity. Untested. |
| **The holdout is untouched** | Zero rows with `split == "holdout"` in 2,960; `truncate_before_holdout` + `assert_no_holdout` remove 2025-01-01 → 2026-06-30 at load time, so those dates are absent from memory in every run. One scripted touch, after freezing. |

---

## 9. Where the evidence lives

```
SEARCH_FINDINGS.md          14 sections: every search, in the order run
HANDOFF.md                  §2 P0-P7 · §4 deviations · §5 P8 · §6 P9
results/runs/walkforward/   the ladder + per-arm daily streams (source of truth)
results/runs/capacity/      the K = 1..8 curve
results/runs/regimes/       separation + profiles + labels (F1's evidence)
results/runs/frequency/     Level 2 spectra
results/runs/inference/     DSR, paired bootstrap, Ledoit-Wolf
results/runs/verify/        held-out verification
results/runs/joint/         joint K x rho search
results/runs/lstm_search/   LSTM stage 1/2 + held-out
results/meeting_figures/    M1-M6, presentation-ready
results/registry.csv        2,960 rows, append-only
```

**Module docstrings carry the "why" for individual components** and are worth
reading directly: `backtest/ledger.py` (closed-form trick), `backtest/costs.py`
(the factor-of-two trap), `data/calendar.py` (why not `bdate_range`),
`data/fred.py` (DGS3MO vs DTB3), `models/recurrent.py` (the LSTM's
pre-registration), `optim/cardinality.py` (NSGA's retirement),
`baselines/hmm_gate.py` (the smoothed-posterior trap), `eval/causality.py`
(why C2 has power both ways), `eval/stats.py` (the √252 unit trap),
`reservoir/diagnostics.py` (memory capacity is not a finance result).

---

## 10. P11 additions (2026-08-29) — F1 closed, the holdout spent

### 10.1 ⭐ F1 closed — the gate was handed VIX and still tracked direction

The gate in F1 read the reservoir state alone, so "it is blind to risk" could
have been an input-starvation artefact rather than a finding. Re-run with the
gate reading `[state ; signals]` (the 3a routing; experts unchanged), K=3,
3 seeds, both universes, alongside a state-only control trained in the same run.

| η², % | sector11 state → signal | hmm | etf10 state → signal | hmm |
|---|---|---|---|---|
| **recent direction (5d)** | **12.5 → 18.4** | 0.2 | **24.8 → 18.4** | 0.1 |
| **realised volatility** | 0.8 → 1.6 | **44.3** | 0.3 → 1.1 | **25.2** |
| short rate | 0.5 → 8.0 | 2.7 | 0.8 → 9.3 | 1.9 |

**Direction stays rank 1; realised volatility stays on the floor**, still ~25×
below the HMM. AMI to the statistical gates stays ≈ 0 (0.007–0.022) and dwell
stays fast (1.8→1.9 wks), so the orthogonality *survives* equal information.

**What it upgrades.** F1 becomes a **revealed preference of the decision
objective** rather than a statement about what the gate could see, and RQ1's
"equal information" clause — which the state-only comparison quietly violated —
is now satisfied.

⚠️ **Anchor the claim on realised volatility, not on the VIX η².** The vix figure
rises (1.4 → 5.9) partly because vix is now literally an input; realised
volatility is computed from returns independently of that channel and does not
move. ⚠️ The gate *does* use the signals (AMI state-vs-signal 0.165 / 0.180) —
but what it takes up is a slow macro **level** (short rate), not risk.

Evidence: `results/runs/regimes_signals/`, figure `M3b_gate_with_signals.png`.

### 10.2 ⭐ The mechanism behind F1 — an odd/even symmetry (exact, three lines)

With `bias_scale = 0` and `x_0 = 0`, the reservoir state is an **exactly odd**
functional of the driving returns: flip every input's sign and the state flips
sign (induction, using that tanh is odd; the trailing z-score preserves it).

**Direction is odd; volatility is even.** A non-degenerate *linear* gate on an odd
state cannot produce a partition invariant under `x → −x`, so **it cannot form a
volatility regime at all.** This predicts the measured up/sideways/down profile
exactly, and explains the HMM contrast: the HMM's observations *contain* trailing
21-day realised volatility, an even feature handed to it ready-made.

Supporting quantification: linearised, the ring's modes span memories of
**1.2–32.8 days**, reproducing §6.7's own numbers from first principles; the tanh
gain at pre-activation std 0.7 is 0.73, shortening the slowest mode to ≈9 days,
close to the **measured** 7.7–8.5 day gate memory. Roughly **90% of the state is
the last 3–4 weeks and 50% the last week**. Two corollaries: leaking is what
*creates* the spread of timescales (at α=1 every mode has the same τ), and **α,
not ρ, is the binding lever on memory** — which explains why the ρ searches found
nothing.

⚠️ **Narrows an existing caveat.** `SEARCH_FINDINGS.md` says the reservoir encodes
both direction and volatility so organising on direction is "a CHOICE". A
*nonlinear* readout could recover even functions; the **linear** gate used here
cannot. Restate accordingly.

### 10.3 The holdout — the null confirmed out of sample (SPENT)

Evaluated once, 2026-08-17, on 2025-01-02 → 2026-06-30 (373 days, 78 weekly
decisions), after the configuration was frozen. **It cannot be run again.**

| | frozen config | best baseline | verdict |
|---|---|---|---|
| sector11 | **0.811** ±0.033 (K=1) | 1/N **0.812** | tied |
| etf10 | **1.295** ±0.023 (K=3) | 1/N **1.340** | tied |

The frozen configuration lands **statistically on top of 1/N** on both universes
— the ideal outcome for a null-result thesis, because the untouched test *agrees*
with the body of evidence. It also **vindicates the freeze**: the labelled
best-found secondary was no better on sector11 and much worse and unstable on
etf10 (0.930 ±**0.583**, turnover 14.6).

⚠️ **The window was benign** — Sharpes are high for *everything* (1/N at 1.34), so
this is not a stress test. ⚠️ One short path: treat frozen and 1/N as **tied**,
never ranked. Evidence: `results/runs/holdout/`.

### 10.4 F4 reinforced — a 1,485-cell grid, and selection still points the wrong way

A dense local grid around each best-capacity optimum (810 + 675 cells), validated
on the selection folds and the top cells re-scored on the full test split.

* **No transferable improvement exists.** etf10's best test config *is* the one
  already in use; sector11's apparent +0.027 is cherry-picked from 12 test
  evaluations with sds of 0.02–0.06.
* **Validation points the wrong way at high resolution**: the val winner
  transferred to 0.522 (sector11) and **0.250** (etf10, from val 0.659).
* **Finer search only inflated validation** — ceiling 0.559 → 0.678, test flat.

⭐ The picture: on the N_r × ρ slice the **val argmax (330, 0.92) and the test
argmax (270, 0.89) sit in different corners**, and the rows nearly swap — the
validation-*coldest* row is the test-hottest. `heatmap_val_vs_test_sector11.png`.

**Write-up value:** the freeze becomes *"~1,500 configurations found nothing that
beats the pre-specified defaults out of sample"* — a result, not a fallback.
Report as a robustness probe, never as a selection.

### 10.5 Two smaller confirmations

* **VIX in the gate does not pay either.** At each universe's best-capacity
  config, adding the signals to the gate moved test Sharpe **−0.023** (sector11,
  within one paired sd) and −0.207 (etf10, on the known-unreliable baseline). The
  state-only control reproduced the capacity reference exactly, which is what
  makes the deltas trustworthy. Consistent with §10.1: a gate that will not
  reorganise onto risk cannot profit from a risk signal.
* **The two loss penalties re-verified at the frozen config** (etf10, K=3,
  post-anneal-fix, 5 seeds): everything flat within seed noise (0.231–0.248). The
  text may now say *re-verified*, not *measured at P8*. Nuance: on etf10 λ_cvar is
  harmless rather than harmful, and λ_cvar = 5 trims maxDD 22.9 → 21.2% at no
  Sharpe cost — the one place a penalty earned anything.

### 10.6 ⛔ Editorial rule — the deflated Sharpe ratio is not mentioned

User decision, 2026-08-29. The DSR does not appear in the dissertation. **L4 and
L5 above are therefore retired from the narrative** and kept only as a record.

The null is argued instead through: the **paired stationary bootstrap** and the
**Ledoit–Wolf robust Sharpe test** (every learned arm's CI straddles zero against
the baselines), the **held-out verification** (six searched winners, none
transferred), the **fine grid** (§10.4), and the **holdout** (§10.3). This also
moots the open trial-count item in §8 — with no DSR there is no count to state.

### 10.7 ✅ Resolves the η² discrepancy flagged at P10

The docs were right; the old CSVs were seed-0.
`run_regime_interpretation.py` defaults to `--seed 0`, so its CSVs (20.9 / 29.6)
are single-seed, while this file quotes the **3-seed mean (12.5 / 24.8)**. The
P11 re-run at 3 seeds reproduces 12.5 / 24.8 exactly. Nothing is wrong with the
finding — the two artifacts are different quantities. Pick one convention and
label it; `results/runs/regimes_signals/` holds 3-seed means.
