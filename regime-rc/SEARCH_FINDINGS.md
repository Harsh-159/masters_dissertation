# Search findings

A running record of the hyperparameter searches, written by the search
scripts as each round completes so the progression is a record rather
than something reconstructed afterwards.

> ⛔ **The deflated Sharpe ratio is not mentioned in the dissertation**
> (editorial decision, 2026-08-29). This file counts configurations as a
> provenance record only, not as input to a deflation.

Metric throughout: **concatenated validation excess Sharpe** - each
seed's fold validation segments stitched into one continuous daily
stream and scored once, the same way the baselines and the literature
are scored. Averaging per-fold Sharpes is not used: it rebases wealth
each January and differed from this by 0.48 on sector11.

---

## sector11 - round 1 (screen)

_2026-07-27 18:06 | 8 folds x 2 seeds | 30 configurations_


### Concatenated validation Sharpe

|   K |   unconstrained |   K_card=3 |   K_card=5 |   K_card=7 |   K_card=9 |
|----:|----------------:|-----------:|-----------:|-----------:|-----------:|
|   1 |           0.611 |      0.587 |      0.586 |      0.611 |      0.611 |
|   2 |           0.544 |      0.515 |      0.51  |      0.502 |      0.544 |
|   3 |           0.541 |      0.436 |      0.47  |      0.516 |      0.541 |
|   4 |           0.516 |      0.452 |      0.463 |      0.505 |      0.516 |
|   5 |           0.525 |      0.437 |      0.46  |      0.491 |      0.525 |
|   6 |           0.527 |      0.423 |      0.433 |      0.522 |      0.527 |


Seed spread (sd) ranged 0.007-0.119; treat differences smaller than this as unresolved.


**Verdict.** Best cell K=1, K_card=7 at +0.611. Carrying K in [1, 2, 3] to round 2 for replication; at two seeds these differences are screening signal only.

---

## sector11 - round 2 (replicate)

_2026-07-27 18:06 | 8 folds x 5 seeds | 15 configurations_


### Concatenated validation Sharpe

|   K |   unconstrained |   K_card=3 |   K_card=5 |   K_card=7 |   K_card=9 |
|----:|----------------:|-----------:|-----------:|-----------:|-----------:|
|   1 |           0.58  |      0.505 |      0.533 |      0.565 |      0.58  |
|   2 |           0.52  |      0.46  |      0.472 |      0.493 |      0.52  |
|   3 |           0.545 |      0.454 |      0.476 |      0.517 |      0.545 |


Seed spread (sd) ranged 0.013-0.082; treat differences smaller than this as unresolved.


**Verdict.** With [0, 1, 2, 3, 4] seeds the leader is K=1, K_card=9 at +0.580. More seeds buy resolution here; a finer K mesh cannot, since K is integer and already dense.

---

## sector11 - round 3 (confirm)

_2026-07-27 18:06 | 12 folds x 5 seeds | 10 configurations_


### Concatenated validation Sharpe

|   K |   unconstrained |   K_card=3 |   K_card=5 |   K_card=7 |   K_card=9 |
|----:|----------------:|-----------:|-----------:|-----------:|-----------:|
|   1 |           0.67  |      0.546 |      0.588 |      0.637 |      0.663 |
|   3 |           0.639 |      0.507 |      0.532 |      0.588 |      0.631 |


Seed spread (sd) ranged 0.012-0.076; treat differences smaller than this as unresolved.


**Verdict.** On the wider F01-F12 era the choice is K=1, K_card=unconstrained at +0.670 (sd 0.026). This is the value to freeze, subject to the usual caveat that it was selected on validation and still owes a held-out check.

---

## Cross-universe K check — is the K result a training defect?

_Diagnostic run, 8 folds x 3 seeds, concatenated validation Sharpe_

Motivation: K=1 winning on sector11 looked implausible, so the K sweep was
repeated on etf10 — the universe where gate G2 found the experts genuinely DID
specialise. If K>1 helps there under the identical procedure, the training is
sound; if it fails on both, the procedure is at fault.

| universe | K=1 | K=2 | K=3 | K=5 | max pair corr (K=3) | gate entropy (K=3) |
|---|---|---|---|---|---|---|
| **etf10** | 1.062 | 1.044 | 1.064 | 1.066 | **0.379** | 0.973 |
| **sector11** | **0.598** | 0.528 | 0.547 | 0.535 | **0.691** | 0.805 |

**Verdict: not a training defect.** The specialisation mechanism demonstrably
works where the universe supports it — on etf10 the experts differentiate
strongly (pairwise correlation 0.17 at K=2, 0.38 at K=3, versus 0.63-0.84 on
sector11), reproducing G2's split verdict exactly. The code does what it was
designed to do.

What it does NOT do is convert that differentiation into out-of-sample value.
On etf10 every K is statistically tied (1.04-1.07 against seed spreads of
0.03-0.13); on sector11, where the experts never differentiate, extra experts
cost performance through capacity alone. So the honest reading is that the
mixture is a working mechanism whose value proposition does not materialise at
this data size on either universe - consistent with the conditioning edge
measuring approximately zero on both.

Separately, a genuine implementation flaw was found and is NOT the cause: R2's
gate-noise anneal targets half of `max_epochs` (epoch 250) while early stopping
fires at epoch 7-30, so every mixture ever trained ran under ~90% of full gate
noise throughout. Removing the noise changed the result by 0.001, but the
schedule is still wrong and should be fixed.

---

## Closing the gap: does lambda_div let K>1 overtake K=1?

_sector11, 8 folds x 3 seeds, concatenated validation Sharpe_

The K sweep ran at lambda_div = 0, so K>1 was never given the tool that forces
specialisation - and sector11's experts are precisely the ones that fail to
specialise unaided. This closes that gap.

| K | lambda_div | Sharpe | ±sd | max pair corr |
|---|---|---|---|---|
| **1** | 0.00 | **0.598** | 0.025 | n/a |
| 3 | 0.00 | 0.547 | 0.026 | 0.691 |
| 3 | 0.01 | 0.534 | 0.030 | 0.677 |
| 3 | 0.05 | 0.420 | 0.066 | 0.487 |
| 3 | 0.20 | 0.382 | 0.084 | 0.400 |

**Verdict: the null hardens.** The penalty does what it is built to do - pairwise
correlation falls monotonically from 0.691 to 0.400 - and performance falls
monotonically WITH it, from 0.547 to 0.382. Forcing the experts apart makes the
mixture strictly worse, and at no setting does K=3 approach K=1's 0.598.

So the earlier conclusion stands and is now stronger than before: on sector11 it
is not that the experts merely fail to specialise, it is that specialisation is
actively counterproductive. Every mechanism that separates them - more experts,
a divergence penalty, or both - costs performance. That is what a genuinely
single-regime universe looks like from the inside.

---

## Reservoir grid RE-RUN at K = 1 (correcting a contaminated selection)

_sector11, 8 folds x 3 seeds, concatenated validation Sharpe_

The first reservoir grid was run at K = 3, but the K search then selected K = 1.
Two contaminations followed: `gate_temperature` is a mathematical no-op at K = 1
(softmax over a single logit is 1.0 whatever the temperature), so the +0.067 the
first grid credited to tau_g = 1.4 could not transfer; and N_r plausibly
interacts with K, since at K = 3 a width of 800 means three experts each reading
800 features.

### Stage 1: N_reservoir x input target_std

| N_r | std=0.7 | std=1.1 | std=1.5 |
|---|---|---|---|
| 150 | 0.543 | 0.507 | 0.445 |
| **300** | **0.598** | 0.516 | 0.519 |
| 500 | 0.541 | 0.534 | 0.510 |
| 800 | 0.548 | 0.582 | 0.554 |
| 1200 | 0.568 | 0.582 | 0.548 |

### Stage 2: rho / alpha at the winner

| variation | Sharpe |
|---|---|
| reference (rho 0.90, alpha 0.30) | 0.598 |
| rho = 0.70 | 0.459 |
| **rho = 0.95** | **0.615** |
| alpha = 0.90 | 0.474 |

**Verdict: the K = 3 grid's answer does not survive.** At K = 1 the optimum moves
from (N_r 800, std 1.10) back to essentially the DEFAULTS - N_r 300, std 0.70 -
with only a marginal rho adjustment to 0.95 (0.615 vs 0.598, against a seed
spread of 0.013). The earlier "+0.28 over default" was an artefact of searching
the reservoir at the wrong K.

Both flat parameters also reassert themselves at K = 1: rho = 0.70 costs 0.14 and
alpha = 0.90 costs 0.12, so their apparent flatness in the random search was a
property of that sample, not of the model.

## Seed ensembling: averaging the portfolios, not the Sharpes

_sector11, K = 1, 8 folds, 5 seeds_

| | Sharpe | turnover | maxDD |
|---|---|---|---|
| mean of individual seeds | 0.580 (sd 0.031) | 16.45 | -0.488 |
| best individual seed | 0.621 | - | - |
| **ensemble (averaged weights)** | **0.585** | 15.92 | -0.487 |

**Verdict: essentially no gain.** +0.005 over the mean seed, -0.036 against the
best seed, and only a 3% turnover reduction. The ensemble is a legitimate and
slightly more stable object than any single seed, but it is not a source of
performance here: the seeds are already tightly clustered (sd 0.031), so there
is little independent noise left for averaging to cancel.

---

## Held-out verification of the reservoir freeze — the winner does not transfer

_sector11, 8 selection folds (F01-F08) vs 6 untouched folds (F13-F18), 5 seeds_

Every reservoir configuration in this project had been chosen by maximising
validation Sharpe on folds F01-F12. Folds F13-F18 (validation years 2018-2023)
were never touched by any search, grid or K sweep, so they are a genuine check on
the *selection process itself*. `scripts/run_verify_frozen.py` runs it.

| config | N_r / rho / alpha / std | selection F01-F08 | held-out F13-F18 |
|---|---|---|---|
| old frozen (random search) | 479 / 0.72 / 0.96 / 0.90 | 0.479 (sd 0.096) | **0.590** (sd 0.030) |
| grid winner | 300 / 0.95 / 0.30 / 0.70 | **0.603** (sd 0.020) | 0.526 (sd 0.085) |
| plan defaults | 300 / 0.90 / 0.30 / 0.70 | 0.580 (sd 0.031) | 0.534 (sd 0.023) |

**Verdict: the selection-fold advantage does not transfer.** The ordering flips,
but the honest statement is stronger and simpler than a reversal: out of sample
the spread ACROSS configurations (0.064) is smaller than the seed spread WITHIN
one configuration (0.085). Which reservoir is chosen matters less than which
random seed is drawn.

Note the trap this creates. `old frozen` tops the held-out column, and adopting it
for that reason would spend F13-F18 as a selection set and leave nothing clean.
The defensible choice is the configuration that required looking at no data at
all — the plan's Appendix B defaults.

---

## Joint K x spectral-radius search — closing the last conditional

_sector11, 12 cells, 8 folds x 3 seeds, concatenated validation Sharpe_

Both axes had been searched, but never together: the K search ran at the default
reservoir, and the reservoir grid ran at a fixed K. Each answer was therefore
conditional on a value the other search later moved. `scripts/run_joint_search.py`
closes that.

### Selection folds F01-F08

| K | rho=0.85 | rho=0.90 | rho=0.95 | rho=0.99 |
|---|---|---|---|---|
| **1** | 0.520 | 0.543 | 0.575 | **0.582** |
| 2 | 0.512 | 0.522 | 0.527 | 0.527 |
| 3 | 0.524 | 0.525 | 0.523 | 0.519 |

**There is a real K x rho interaction, and it explains an earlier contradiction.**
At K = 1 the spectral radius matters and is monotone increasing (0.520 -> 0.582,
a swing of 0.062). At K = 3 it is completely flat (0.524 -> 0.519, a swing of
0.006). So the first grid, run at K = 3, was right that rho looked flat *in that
sample*, and the K = 1 re-run was right that it matters — the two are not in
conflict, they are two slices of an interaction. With three experts sharing the
routing, the memory depth of any one reservoir stops being the binding constraint.

### Held-out F13-F18

| config | held-out Sharpe |
|---|---|
| search winner: K=1 rho=0.99 | **0.428** |
| defaults: K=1 rho=0.90 | 0.509 |
| defaults: K=3 rho=0.90 | 0.518 |

**Verdict: the winner does not transfer, for the fourth time.** It scores 0.081
BELOW the defaults it was selected to beat. The pre-registered rule stated before
the run — adopt only if it survives the held-out check — therefore rejects it.

Note also that K = 3 edges K = 1 on the held-out era (0.518 vs 0.509). That gap is
far inside the seed noise, so it does not overturn the K = 1 selection, but it is
worth stating: **the K = 1 result was established on 2006-2013 and is weaker on
2018-2023**, which is consistent with the params-per-decision argument (the
training window is roughly twice as long by F13).

### The cumulative picture

Four hyperparameter selections have now been made on validation and checked:

1. first random search: 0.817 -> 0.298 across seeds;
2. first reservoir grid: did not survive the move from K = 3 to K = 1;
3. K = 1 reservoir grid: 0.603 selection -> 0.526 held out (worst of three);
4. joint K x rho search: 0.582 selection -> 0.428 held out (worst of three).

**Not one has transferred.** That is no longer a run of bad luck; it is the
finding. On this problem, at this data size, reservoir hyperparameter selection
on validation folds does not generalise, and the pre-specified defaults are the
honest configuration to report.

---

## etf10 — closing the gap: the universe that had never been verified

_etf10, 5 selection folds (F01-F05) vs 5 untouched folds (F06-F10)_

Until this round, every "the defaults are optimal" statement rested on sector11
evidence alone. etf10 had had **one** search (the 50-draw random search on
F01-F05) and **no** verification of any kind, yet its random-search winner was
still sitting in `configs/frozen/etf10.yaml`. Asserting the defaults were right
there was unsupported. This closes it.

### Held-out verification (5 seeds, at K = 3)

| config | selection F01-F05 | held-out F06-F10 |
|---|---|---|
| random-search winner (N_r 415, rho 0.826, alpha 0.310, std 1.14) | 0.438 (sd 0.071) | **0.929** (sd 0.150) |
| plan defaults (N_r 300, rho 0.90, alpha 0.30, std 0.70) | **0.474** (sd 0.050) | 0.811 |

Held-out difference **+0.118 against a seed spread of 0.150** - within noise.

Two things worth recording. First, the searched config is *worse on the very
folds it was selected on* (0.438 vs 0.474), because it was originally scored
under a different training budget (200 epochs / patience 15) and before the
gate-noise anneal fix; its recorded `val_sharpe: 1.0616` is not comparable to
any current number. Second, and unlike sector11, the direction here **favours**
the searched config. So etf10 is "no evidence it transfers", NOT "evidence it
fails". The two universes should not be flattened together.

### Joint K x spectral-radius grid (5 folds x 3 seeds)

| K | rho=0.85 | rho=0.90 | rho=0.95 | rho=0.99 |
|---|---|---|---|---|
| 1 | 0.399 | 0.401 | 0.403 | 0.405 |
| **2** | 0.434 | 0.431 | 0.428 | **0.464** |
| 3 | 0.409 | 0.412 | 0.413 | 0.415 |

**The K x rho interaction does NOT replicate on etf10.** On sector11, rho swung
0.062 at K = 1 and only 0.006 at K = 3 - a clear interaction. Here rho is flat at
every K (swings 0.006-0.036). So the interaction is a **sector11 phenomenon**
and must be reported as such, not as a general property of the architecture.

A third K answer also appears: sector11 selected K = 1, the plan defaults to
K = 3, and etf10's own search prefers K = 2. All three are inside seed noise,
which is itself the point.

### Held out

| config | held-out F06-F10 |
|---|---|
| search winner K=2 rho=0.99 | 0.783 |
| **defaults K=1 rho=0.90** | **0.809** |
| defaults K=3 rho=0.90 | 0.678 |

**The winner does not transfer** (-0.027 against K = 1 defaults).

### The count is now five

| # | universe | selection | validation | held out |
|---|---|---|---|---|
| 1 | sector11 | first random search | 0.817 | 0.298 across seeds |
| 2 | sector11 | first grid (at K = 3) | won | died on the move to K = 1 |
| 3 | sector11 | K = 1 reservoir grid | 0.603 | 0.526 - worst of three |
| 4 | sector11 | joint K x rho | 0.582 | 0.428 - worst of three |
| 5 | **etf10** | joint K x rho | 0.464 | 0.783 vs defaults 0.809 |

**Five selections, two universes, none transferred.** The pre-specified
Appendix B defaults are frozen on both, and the hyperparameter search is
reported as a negative result - now on evidence from both testbeds rather than
one.

---

## LSTM hyperparameter search — closing the D2 fairness gap

_sector11 (8 selection folds) and etf10 (5), 2 screening seeds; held-out check at 3_

Decision D2 claims a frozen encoder beats a trained one. Until now that rested on
an unfair comparison: **238 search trials on the reservoir side against 0 for the
LSTM**, which ran at one hand-picked configuration (hidden 32, lookback 60).
`scripts/run_lstm_search.py` gives it the same protocol - same folds, same seeds,
same concatenated metric, same held-out rule.

### Stage 1: capacity x memory horizon

**sector11**

| hidden \ lookback | 20 | 60 | 120 | 250 |
|---|---|---|---|---|
| 16 | 0.419 | 0.402 | 0.402 | 0.402 |
| 32 | 0.528 | 0.454 | 0.495 | 0.442 |
| 64 | 0.803 | **0.904** | 0.817 | 0.777 |
| 128 | **0.953** | 0.860 | 0.824 | 0.812 |

**etf10**

| hidden \ lookback | 20 | 60 | 120 | 250 |
|---|---|---|---|---|
| 16 | 0.652 | 0.706 | 0.686 | 0.720 |
| 32 | 0.871 | 0.856 | 0.859 | 0.862 |
| 64 | 0.991 | 0.945 | 0.927 | 0.950 |
| 128 | **1.068** | 1.039 | 1.022 | 1.022 |

**Two findings, both clean and both replicated across universes.**

**1. Capacity matters enormously, and the original choice of 32 was a real
handicap.** Averaged over lookback, hidden 16 -> 128 moves 0.406 -> 0.862 on
sector11 and 0.691 -> 1.038 on etf10 - swings of +0.456 and +0.347 against
median seed spreads of 0.036 and 0.090. **This contradicts the argument that
fixed hidden=32 in the first place** ("a fold holds only a few hundred weekly
decisions; a large network would lose for the wrong reason"). Measured, the
opposite is true on validation: more capacity, monotonically better.

**2. The memory horizon does not matter, which settles the question directly.**
Averaged over capacity, lookback 20 -> 250 swings only +0.067 (sector11) and
+0.022 (etf10). The capacity effect is **6.8x and 15.5x** the lookback effect.
If anything SHORTER windows are better. So the LSTM's 60-day receptive field was
never the binding constraint - corroborating the measured reservoir memory
capacity of 19.53 with 69% of it inside 60 days. **"You crippled the LSTM by
giving it bounded memory" is now answerable with a number: no, and here it is.**

### Stage 2 (one-at-a-time at the stage-1 winner)

| variation | sector11 | etf10 |
|---|---|---|
| stage-1 winner (hidden 128, lookback 20) | 0.953 | **1.068** |
| num_layers = 2 | 0.675 | 0.912 |
| dropout = 0.2 | 0.953 | 1.068 |
| learning_rate = 0.001 | 0.596 | 0.982 |
| learning_rate = 0.01 | **1.011** | 1.002 |

Depth hurts on both. Dropout is exactly inert (it is a no-op at one layer, as
expected). Learning rate splits: 0.01 helps sector11, hurts etf10.

### Held out — and the search fails to transfer, again

| universe | search champion | ladder baseline | difference | seed spread |
|---|---|---|---|---|
| sector11 | 0.738 (sd 0.220) | 0.554 (sd 0.036) | **+0.183** | 0.220 |
| etf10 | 0.473 (sd 0.037) | 0.605 (sd 0.065) | **-0.132** | 0.065 |

Neither adoption is supportable. On sector11 the champion leads but by less than
its own seed spread; on etf10 it is clearly WORSE, by roughly two seed spreads.
The pre-registered rule therefore rejects it on both.

Note also that the high-capacity champion is markedly less seed-stable
(sd 0.220 against the baseline's 0.036 on sector11) - more capacity buys
validation performance and pays for it in variance, which is the same trade the
K sweep found.

### The count is now six, and it covers both encoders

| # | universe | encoder | selected on validation | transferred? |
|---|---|---|---|---|
| 1-4 | sector11 | reservoir | random / grid / K=1 grid / joint | no |
| 5 | etf10 | reservoir | joint K x rho | no |
| 6 | **both** | **LSTM** | hidden 128, lookback 20 (+ lr) | **no** |

**The finding generalises.** It is no longer "reservoir hyperparameters do not
transfer" - it is **hyperparameter selection does not transfer for either
encoder at this data size**. That is a statement about the problem, not about
reservoir computing, and it is considerably stronger than what we had.

### What this does to D2

D2 survives, and the comparison is now fair: the trained encoder was given a
matched search across capacity and memory horizon plus depth, dropout and
learning rate, and no configuration it found survived held-out verification.
The ladder keeps hidden 32 / lookback 60 - not because that was well chosen (it
was not) but because adopting the searched winner would break the same rule that
rejected five reservoir winners.

**This must be stated as a limitation, not buried:** on the folds a practitioner
would have searched, hidden=128 is far better than hidden=32, so the LSTM arm in
the ladder is running at a configuration the search says is poor. The defensible
position is that no configuration transferred, so the pre-specified one stands -
but a reader is entitled to know the selection-fold gap was large.

---

## What ARE the learned regimes? Level 1 interpretability

_Both universes, all folds, 3 seeds. `scripts/run_regime_interpretation.py`_

Gate G3 established that decision-defined regimes carve the market DIFFERENTLY
from statistical ones (AMI ~ 0). That says they disagree; it never said what
either scheme IS. Two forecasters can both say "there are three kinds of day"
and never agree, and the AMI cannot tell you that one tracks temperature and the
other humidity. This closes that gap.

**Method.** Each week already carries a regime label from each scheme. Describe
the market that week with causal state variables, then compute (a) eta squared -
the fraction of each variable explained by the labelling - and (b) the
standardised mean of each variable within each regime.

### Separation: what each scheme tracks (eta squared, %, mean of 3 seeds)

**sector11**

| variable | learned | HMM | VIX |
|---|---|---|---|
| realised volatility | 0.8 | **44.3** | 18.9 |
| dispersion | 0.6 | **36.2** | 11.0 |
| VIX level | 1.3 | 20.7 | **65.4** |
| credit spread | 0.4 | 20.7 | 28.0 |
| momentum 63d | 1.7 | 14.8 | 22.6 |
| avg correlation | 0.8 | 12.1 | 5.9 |
| **recent direction (5d)** | **12.5** | **0.2** | **4.5** |

**etf10**

| variable | learned | HMM | VIX |
|---|---|---|---|
| realised volatility | 0.3 | **25.2** | 11.3 |
| dispersion | 0.2 | **19.0** | 15.7 |
| VIX level | 3.5 | 2.5 | **62.8** |
| momentum 63d | 2.9 | 11.1 | 17.3 |
| **recent direction (5d)** | **24.8** | **0.1** | **3.2** |

**The finding, in one line: the statistical gates are RISK detectors and the
learned gate is a DIRECTION detector.** Each is nearly blind to the other's
variable - the HMM explains 0.1-0.2% of recent direction, the learned gate
0.3-0.8% of volatility.

`return_5d` ranks FIRST for the learned gate in all six runs (3.3% to 29.6%),
and the runner-up never exceeds 5.2%. The magnitude is seed-sensitive; the
identity of the variable is not.

### What the regimes mean (learned gate, standardised means)

|  | regime 0 | regime 1 | regime 2 |
|---|---|---|---|
| sector11, recent direction | +0.57 | -0.51 | -0.24 |
| etf10, recent direction | +0.54 | -0.12 | -0.70 |

Three states meaning *the market just rose / sideways / just fell*. Not
calm-versus-stressed.

### This explains two earlier observations

**AMI ~ 0 now has a mechanism.** The schemes are not disagreeing about where to
draw a line on one axis; they are measuring different axes. Corroborating: the
two RISK detectors agree with each other more than either agrees with the
learned gate (HMM vs VIX AMI +0.093 / +0.017, against ~0.00 for either vs
learned).

**The fast switching is a consequence, not a quirk.**

| scheme | mean dwell | switch rate |
|---|---|---|
| learned | 1.7-1.8 wks | 56-59% |
| VIX | 3.3-3.7 wks | 27-30% |
| HMM | 9.2-12.9 wks | 8-11% |

Volatility persists, so a scheme tracking it sits still for 9-13 weeks.
Five-day direction flips constantly, so a scheme tracking it must switch
constantly. The dwell time follows from the variable.

### Caveats, stated

1. **This runs at K = 3, but sector11's frozen configuration is K = 1**, where
   the gate is mathematically inert and there are NO regimes to interpret. We
   are interpreting the regime structure of a configuration that was not
   selected. This must be said plainly in the write-up.
2. **The reservoir encodes both direction and volatility**, so organising on
   direction is a CHOICE the decision objective made, not the only option it
   had. That is what makes it a finding.
3. sector11 seed 1 is weak (3.3%); the variable identity is stable, the
   strength is not.

### Why this matters

It converts the RQ1 null into a positive, mechanistic claim:

> Decision-defined regimes are not a noisy version of statistical regimes. They
> are orthogonal because they measure something else. Statistical detection
> finds RISK states (volatility, dispersion, correlation); the decision
> objective, given the same reservoir state and free to organise however it
> likes, finds DIRECTION states, switching five to seven times faster as a
> direct consequence. Holds on both universes and in every seed.

This stands independently of the performance null, and it sets up the Level-2
frequency reading: direction is high-frequency structure, volatility is
low-frequency, which is the Fourier decomposition the cycle reservoir performs
(Fong, Li and Tino, Chaos 2025).

---

## The capacity curve: K = 1 to 8, each K given its own reservoir search

_Both universes. Inner search 9 reservoir cells x selection folds x 2 seeds;
outer score on the FULL walk-forward test split x 3 seeds._
_`scripts/run_capacity_curve.py`, figure F10._

### Why this supersedes the earlier K search

The original K search (three rounds, K = 1 selected) ran every K at ONE reservoir
configuration. That judges each K at settings tuned for a competitor, and the
joint K x rho grid had already shown the two interact. This run searches the
reservoir **inside each K**, so every capacity is compared at its own best
settings.

Scope: plain `MixtureOfExperts` only - K experts and a learned gate reading the
frozen reservoir state. No market signals (rungs 3a/3b/3c), no statistical
routing (HMM, VIX). Mixing an information change into a capacity sweep would
make the curve uninterpretable.

### sector11 (18 folds)

| K | best N_r | best rho | validation | **TEST** | sd | params | gap |
|---|---|---|---|---|---|---|---|
| 1 | 300 | 0.90 | 0.611 | **0.403** | 0.002 | 3,913 | 0.208 |
| **2** | 300 | 0.85 | 0.559 | **0.505** | 0.031 | 7,826 | **0.055** |
| 3 | 150 | 0.90 | 0.563 | 0.474 | 0.042 | 5,889 | 0.089 |
| 4 | 600 | 0.85 | **0.615** | 0.485 | 0.041 | 31,252 | 0.130 |
| 5 | 300 | 0.85 | 0.564 | 0.459 | 0.019 | 19,565 | 0.105 |
| 6 | 600 | 0.90 | 0.613 | 0.439 | 0.019 | 46,878 | 0.175 |
| 7 | 600 | 0.95 | 0.578 | 0.455 | 0.021 | 54,691 | 0.122 |
| 8 | 600 | 0.95 | 0.561 | 0.454 | 0.024 | 62,504 | 0.107 |

### etf10 (10 folds)

| K | best N_r | best rho | validation | **TEST** | sd | params | gap |
|---|---|---|---|---|---|---|---|
| 1 | 600 | 0.95 | 0.503 | 0.274 | 0.019 | 7,212 | 0.230 |
| 2 | 600 | 0.85 | 0.531 | 0.349 | 0.144 | 14,424 | 0.182 |
| 3 | 300 | 0.90 | 0.477 | 0.249 | 0.025 | 10,836 | 0.228 |
| 4 | 150 | 0.95 | 0.475 | 0.264 | 0.036 | 7,248 | 0.211 |
| **5** | 600 | 0.95 | **0.636** | **0.411** | 0.145 | 36,060 | 0.225 |
| 6 | 600 | 0.95 | 0.531 | 0.314 | 0.155 | 43,272 | 0.217 |
| 7 | 600 | 0.85 | 0.555 | 0.215 | 0.007 | 50,484 | 0.340 |
| 8 | 600 | 0.95 | 0.542 | 0.292 | 0.147 | 57,696 | 0.250 |

### Four findings

**1. K = 1 is the WORST capacity on sector11 test, not the best.** This
CORRECTS the earlier K search. Against K = 1, and resolved at more than two seed
spreads: K = 2 (+0.102), K = 5 (+0.056), K = 7 (+0.052), K = 8 (+0.051). Four of
seven capacities beat it. The earlier K = 1 selection was an artefact of
comparing capacities at a reservoir tuned for one of them.

**2. On sector11, validation and test ANTI-CORRELATE across K: r = -0.477.**
K = 4 has the highest validation of all (0.615) and mid-table test; K = 2 has
near-worst validation (0.559) and the best test (0.505). Selecting K on
validation does not merely fail to help here - it points the wrong way. This is
the sharpest single piece of evidence in the project for the transfer-failure
result. Note it does NOT generalise: on etf10 the correlation is +0.678.

**3. K and the reservoir genuinely interact.** The searched width climbs with K
on sector11 (300, 300, 150, 600, 300, 600, 600, 600). Any K comparison at a
fixed reservoir was measuring the wrong thing.

**4. None of it beats the baselines.** Best K on sector11 is 0.505 against
inverse-vol's 0.511; best K on etf10 is 0.411 against SPY's 0.688. Eight
capacities, each individually tuned, and the performance null is untouched.

### What this does NOT license

The frozen sector11 configuration stays at K = 1 even though the curve says
K = 2 is better on test. Re-selecting on the basis of a test-set curve is
precisely the selection bias this project spent six held-out checks avoiding.
The curve is reported as a finding; the frozen configuration is not changed, and
the write-up says so explicitly.

### Caveat

etf10's seed spreads reach 0.155, so only K = 7 resolves there (and negatively).
sector11's are 0.002-0.042, so its conclusions are much firmer. The two
universes should not be pooled.

---

## Level 2: does the gate exploit the reservoir's frequency decomposition?

_Both universes, all folds, 3 seeds. `scripts/run_frequency_analysis.py`._

**A prediction was registered before this ran, and it was wrong.** That is the
main reason the section is here.

### The theory, and why it applies exactly

The Simple Cycle Reservoir wires its units into a ring. With leaking, the
linearised state update is

    x_t = A x_{t-1} + alpha * W_in u_t ,   A = (1 - alpha) I + alpha * rho * P

where P is the cyclic shift. **A is circulant**, so it is diagonalised exactly by
the discrete Fourier basis - and it stays circulant under leaking, which is the
part that makes this legitimate rather than an approximation of the recurrence.
This is the structure behind Fong, Li and Tino (Chaos 2025): a linear cycle
reservoir performs a Fourier decomposition of its driving signal.

Each mode k has eigenvalue `lambda_k = (1-alpha) + alpha*rho*exp(-2i*pi*k/N)` and
therefore a memory timescale `tau_k = -1/ln|lambda_k|` in days. At the frozen
configuration (N=300, rho=0.90, alpha=0.30) the 151 independent modes span:

| mode k | 0 | 50 | 100 | 150 |
|---|---|---|---|---|
| memory tau (days) | 32.8 | 7.0 | 2.0 | 1.2 |

So the reservoir is not an opaque bag of 300 numbers. **It is a bank of 300
filters with known memory lengths, and a linear readout chooses which
timescales to listen to.**

### The pre-registered prediction

Level 1 found the learned gate organises its regimes around 5-day return
direction while the HMM organises around volatility (dwell 9-13 weeks). If that
is right, the gate should concentrate its weight on the SHORT-memory modes.

Stated in the module docstring before the run: **the gate's spectral mass should
concentrate at small tau.**

### The result: it does not

Three quantities, so the reading does not rest on the linearisation alone -
the gate's weight power (which modes it points at), the state power (which modes
actually carry signal, measured on real cached states), and their product (what
the gate logit's variance is actually made of).

| % of power | trained gate | untrained gate | state | contribution |
|---|---|---|---|---|
| **sector11** | | | | |
| fast (< 3 days) | 42.6 | 45.6 | 30.8 | 28.3 |
| medium (3-10 days) | 28.5 | 28.5 | 27.5 | 25.4 |
| slow (> 10 days) | 28.9 | 25.9 | 41.7 | 46.3 |
| weighted-average tau | **8.5 d** | **7.7 d** | 11.2 d | 11.9 d |
| **etf10** | | | | |
| fast (< 3 days) | 44.4 | 45.3 | 28.4 | 27.0 |
| medium | 27.3 | 27.2 | 28.2 | 26.3 |
| slow (> 10 days) | 28.4 | 27.4 | 43.4 | 46.7 |
| weighted-average tau | **8.4 d** | **8.1 d** | 11.7 d | 12.1 d |

**The trained gate is statistically indistinguishable from random
initialisation.** Weighted-average memory moves 7.7 -> 8.5 days on sector11 and
8.1 -> 8.4 on etf10. On etf10 the largest per-mode power shift (0.0032) is
SMALLER than a flat spectrum's own bin height (0.0066).

The null used is the same architecture at the same seed with no gradient steps -
a random readout has flat expected power, so any structure in the trained
spectrum would be learned.

### A second small-sample false positive, worth keeping

**On 2 folds this looked dramatic and reversed.** The smoke test showed the
trained gate fleeing the fast modes (25.9% against the untrained 45.6%) - a
clean, publishable-looking effect in the OPPOSITE direction to the prediction.
On all 18 folds it evaporated to 42.6% against 45.6%.

Reporting the two-fold version would have been a confident wrong answer. This is
the second such instance in the project, after the first random search froze a
configuration at 0.817 that scored 0.298 across seeds, and it belongs beside it
in the methods chapter: **small samples do not merely add noise, they
manufacture findings.**

### What the null actually tells us

1. **It constrains Level 1.** The gate tracks 5-day direction (robust, six
   runs). It does NOT achieve that by tuning into a frequency band. Whatever the
   decision objective learned is not expressible as timescale selection in the
   readout.
2. **The slow-weighted effective read is the reservoir's doing, not the
   gate's.** Effective memory comes out at ~12 days, but that is because the
   STATE concentrates 42-43% of its power in slow modes while the gate reads
   roughly uniformly. The reservoir's dynamics set the timescale; the readout
   accepts it.
3. **It is a real constraint on the SCR story.** The Chaos 2025 result says the
   reservoir PROVIDES a frequency decomposition. This measurement says a
   decision-trained readout does not exploit it selectively - it reads broadly
   and lets the reservoir's own spectral shaping do the work.

### Caveat

The Fourier diagonalisation is exact for the LINEAR system; this reservoir has
tanh. The input scale is calibrated to a pre-activation std of 0.7, the
mildly-curved rather than saturated part, so a first-order reading is
defensible - and the empirical state spectrum is included precisely so the
conclusion does not rest on the linearisation alone.

**Write-up placement: two paragraphs or an appendix, not a headline.** Level 1
is the material worth presenting; Level 2 is the honest "we checked the obvious
mechanism and it was not that", plus a pre-registered prediction that failed,
which is worth more methodologically than a confirmation would have been.

---

## The dense local grid — 1,485 configurations, and none of them transfer

_Run 2026-08-17. sector11 810 cells, etf10 675 cells. Validation on the selection
folds (2 seeds); the top 12 re-scored on the full walk-forward TEST split
(3 seeds). `scripts/run_fine_grid.py`, output `results/runs/grid_fine/`._

The capacity curve searched a coarse 9-cell reservoir grid inside each K. This
asks whether a **finer** search around each winner finds anything better — the
coarse grid could have stepped over a local optimum. Grids are centred on each
universe's best-capacity config, densest on the two axes with measured effect:

| axis | sector11 (K=2) | etf10 (K=5) |
|---|---|---|
| N_r | 250–350 step 20 (6) | 500–700 step 50 (5) |
| rho | 0.80–0.92 (5) | 0.90–0.99 (5) |
| alpha | 0.20 / 0.30 / 0.40 | 0.20 / 0.30 / 0.40 |
| input std | 0.60 / 0.70 / 0.80 | 0.60 / 0.70 / 0.80 |
| tau_g | 0.80 / 1.00 / 1.20 | 0.80 / 1.00 / 1.20 |

### Top of the validation table, and where each lands on test

| universe | best-validation cell | val | -> test |
|---|---|---|---|
| sector11 | N_r 330, rho 0.83, a 0.30, std 0.80, tau 1.20 | **0.678** | 0.522 |
| sector11 | best TEST among the top 12 (N_r 330, rho 0.92, std 0.70) | 0.633 | **0.532** |
| etf10 | N_r 600, rho 0.99, a 0.20, std 0.60, tau 1.20 | **0.659** | **0.250** |
| etf10 | best TEST among the top 12 | 0.636 | **0.411** (= the capacity config) |

Capacity-curve references: sector11 0.505, etf10 0.411.

### Three findings

1. **No transferable improvement exists.** etf10's best test cell *is* the config
   already in use. sector11's apparent +0.027 over the reference is picked from
   12 test evaluations whose seed sds are 0.02-0.06, and 6 of the top 12 land
   *below* the reference - noise around it, not an improvement.
2. **Validation points the wrong way, at high resolution.** The validation winner
   transfers to mid-table on sector11 and badly on etf10 (0.659 -> 0.250).
   Selecting off this grid would produce a configuration **worse than the
   untuned defaults**.
3. **The extra resolution only inflated validation.** The validation ceiling rose
   0.559 -> 0.678 while test stayed flat: more cells means a higher maximum of a
   noisy quantity, i.e. more overfitting of the selection metric, and no more
   signal.

### The picture: validation-hot is not test-hot

`make_grid_fine_heatmap.py` draws the N_r x rho slice at default nuisances
(alpha 0.30, std 0.70, tau 1.00), computing test on the *same* cells as val:

* **validation argmax (N_r 330, rho 0.92) = 0.636**
* **test argmax (N_r 270, rho 0.89) = 0.520**

They sit in different corners, and the rows nearly swap: N_r 330 is
validation-hot but test-mediocre, while N_r 270 is among the *coldest* on
validation (0.51-0.53) and the **hottest on test**. The whole test surface spans
only 0.39-0.52. Figure: `heatmap_val_vs_test_sector11.png`.

### What this licenses, and what it does not

It licenses the strong form of the freeze: *a dense grid of ~1,500 configurations
around each best-capacity optimum found no reservoir that beats the pre-specified
defaults out of sample, and the in-sample winner transferred worse than the
defaults.* That is a **result**, not a fallback.

It does **not** license quoting any cell from this grid as a tuned configuration.
This is reported as a robustness probe of the defaults, never as a selection.

⚠️ Caveats: the top-validation cells carry large seed spreads (0.16-0.22 on
sector11), so even the validation ranking is barely resolved; etf10's test sds
(~0.10-0.16) are the known-unreliable ones; and the exact winning cell should not
be over-read - the finding is the **mismatch of the bright regions**, not the
identity of any single cell.
