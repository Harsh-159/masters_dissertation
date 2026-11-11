"""The filtered-HMM statistical gate - the comparison the thesis rests on.

What this is
------------
Rung 4's first statistical gate. A Gaussian hidden Markov model is fitted to
two market observables - the equal-weight universe excess return and its
trailing 21-day realised volatility - and its **filtered** state probabilities
replace the learned gate's memberships. The experts are then retrained from
scratch under that fixed routing (R6), so the comparison against the learned
system is at strictly equal conditions: same loss, same schedule, same seeds,
same training code path.

The trap this module exists to avoid (E6 / A5)
----------------------------------------------
The natural implementation - ``hmmlearn.predict_proba`` - returns
**smoothed** posteriors: probabilities computed by the forward-*backward*
algorithm, which conditions on the *entire* sequence, future included. Handing
those to the baseline would give it information the learned system does not
have, quietly corrupting the one comparison the whole dissertation rests on -
and in whichever direction, nobody would know.

So: ``hmmlearn`` is used only to FIT the parameters, on the training window
only. The probabilities used everywhere are produced by the hand-written
log-space **forward** filter below, which conditions on observations up to and
including day t and nothing later. Two unit tests keep it honest forever:

1. altering any observation after t leaves the filtered probabilities at t
   bit-identical;
2. the filter's output *diverges* from ``predict_proba`` - if the two ever
   matched exactly, the wrong function would be being called somewhere.

Degeneracy rule (the plan's watch-out)
--------------------------------------
If one fitted state occupies more than 95 percent of the training window, the
HMM has effectively collapsed to fewer regimes; the fit is retried with one
state fewer and the change is recorded. The check runs on filtered (causal)
occupancies, not smoothed ones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

logger = logging.getLogger(__name__)

# Observation dimensions: equal-weight excess return, trailing realised vol.
REALISED_VOL_WINDOW = 21

# A state occupying more than this fraction of the training window means the
# model has collapsed to fewer effective regimes.
DEGENERATE_OCCUPANCY = 0.95


@dataclass(frozen=True)
class FittedHmmGate:
    """A fitted HMM gate plus its filtered membership series.

    Attributes
    ----------
    n_states
        The number of states actually used (after any degeneracy reduction).
    filtered
        ``(T, n_states)`` filtered probabilities on the full calendar span the
        observations covered - each row conditions only on data up to its own
        day.
    observation_dates
        The dates the ``filtered`` rows correspond to.
    reduced_from
        The originally requested state count, if degeneracy forced a
        reduction; ``None`` otherwise. Recorded in the registry note.
    """

    n_states: int
    filtered: np.ndarray
    observation_dates: pd.DatetimeIndex
    reduced_from: int | None


def build_observations(
    risky_returns: pd.DataFrame,
    availability: pd.DataFrame,
    risk_free: pd.Series,
) -> pd.DataFrame:
    """The HMM's two observables, built causally on the universe calendar.

    * ``excess_return``: the equal-weight mean, over the assets active that
      day, of daily simple returns in excess of ``r_f`` - the plainest
      "how did the market do today" number.
    * ``realised_vol``: the trailing 21-day standard deviation of that same
      series - "how rough has the last month been". The window ends at the
      current day, so both observables are known at that day's close.

    Returns
    -------
    pandas.DataFrame
        Columns ``excess_return`` and ``realised_vol``; NaN during the initial
        volatility warm-up, which callers must slice past.
    """
    active = availability.to_numpy() == 1
    excess = risky_returns.to_numpy() - risk_free.to_numpy()[:, None]

    with np.errstate(invalid="ignore"):
        equal_weight_excess = np.where(active, excess, np.nan)
        daily = np.nanmean(equal_weight_excess, axis=1)

    observations = pd.DataFrame(index=risky_returns.index)
    observations["excess_return"] = daily
    observations["realised_vol"] = (
        observations["excess_return"].rolling(REALISED_VOL_WINDOW).std(ddof=1)
    )
    return observations


def diagonal_variances(model: GaussianHMM) -> np.ndarray:
    """Extract per-state diagonal variances from a fitted GaussianHMM.

    A known hmmlearn trap: with ``covariance_type="diag"`` the public
    ``covars_`` property inflates the stored ``(K, d)`` variances into full
    ``(K, d, d)`` diagonal matrices. Reshaping that to ``(K, -1)`` silently
    interleaves zeros into the variance vector and every log-density after it
    is wrong. This helper takes the actual diagonals, whichever form appears.
    """
    covariances = np.asarray(model.covars_)
    if covariances.ndim == 3:
        return np.einsum("kii->ki", covariances)
    return covariances


def log_gaussian_diag(
    observations: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    """Log-density of each observation under each state's diagonal Gaussian.

    Parameters
    ----------
    observations
        ``(T, d)`` observations.
    means, variances
        ``(K, d)`` per-state parameters (diagonal covariance).

    Returns
    -------
    numpy.ndarray
        ``(T, K)`` log-densities.
    """
    spread = observations[:, None, :] - means[None, :, :]         # (T, K, d)
    log_density = -0.5 * (
        np.log(2.0 * np.pi * variances)[None, :, :] + spread**2 / variances[None, :, :]
    )
    return log_density.sum(axis=2)


def forward_filter(
    observations: np.ndarray,
    log_start_probabilities: np.ndarray,
    log_transition_matrix: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """The causal forward filter - Appendix C.2, exactly.

    Produces ``p(state_t | observations up to and including t)`` for every t,
    in log space for numerical safety. This is the *filtered* distribution: no
    backward pass, no future observations, ever.

    Returns
    -------
    numpy.ndarray
        ``(T, K)`` rows on the simplex.
    """
    log_likelihoods = log_gaussian_diag(observations, means, variances)

    log_alpha = log_start_probabilities + log_likelihoods[0]
    filtered = [np.exp(log_alpha - logsumexp(log_alpha))]

    for time_index in range(1, len(observations)):
        log_alpha = (
            logsumexp(log_alpha[:, None] + log_transition_matrix, axis=0)
            + log_likelihoods[time_index]
        )
        filtered.append(np.exp(log_alpha - logsumexp(log_alpha)))

    return np.vstack(filtered)


def fit_hmm_gate(
    observations: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    n_states: int,
    random_state: int = 0,
    n_iter: int = 200,
) -> FittedHmmGate:
    """Fit on the training window only; filter forward over the whole span.

    Parameters
    ----------
    observations
        The full-span observable frame from :func:`build_observations`.
    train_dates
        The (embargoed) training window - the ONLY rows the fit may see.
    n_states
        Requested state count (matched to the learned system's K). Reduced by
        one and refitted if a state degenerates to >95 percent occupancy.
    random_state, n_iter
        EM initialisation seed and iteration cap - fixed, recorded, and part
        of the experiment's identity.

    Returns
    -------
    FittedHmmGate
    """
    finite = observations.dropna()
    train_rows = finite.loc[finite.index.intersection(train_dates)]
    if len(train_rows) < 200:
        raise ValueError(
            f"Only {len(train_rows)} finite training observations - too few "
            "to fit an HMM credibly."
        )

    requested = n_states
    while True:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=random_state,
        )
        model.fit(train_rows.to_numpy())

        filtered = forward_filter(
            finite.to_numpy(),
            log_start_probabilities=np.log(model.startprob_ + 1e-300),
            log_transition_matrix=np.log(model.transmat_ + 1e-300),
            means=model.means_,
            variances=diagonal_variances(model),
        )

        # Degeneracy check on the CAUSAL occupancies over the training rows.
        train_positions = finite.index.get_indexer(train_rows.index)
        train_occupancy = filtered[train_positions].mean(axis=0)
        if train_occupancy.max() <= DEGENERATE_OCCUPANCY or n_states <= 2:
            break
        logger.warning(
            "HMM degenerate (one state at %.1f%% occupancy); refitting with "
            "%d states (plan's watch-out rule).",
            100 * train_occupancy.max(), n_states - 1,
        )
        n_states -= 1

    return FittedHmmGate(
        n_states=n_states,
        filtered=filtered,
        observation_dates=finite.index,
        reduced_from=requested if n_states != requested else None,
    )


def memberships_at(
    gate: FittedHmmGate, dates: pd.DatetimeIndex, signal_positions: np.ndarray
) -> np.ndarray:
    """The filtered memberships at given signal days.

    Parameters
    ----------
    gate
        The fitted gate.
    dates
        The universe calendar the signal positions index into.
    signal_positions
        Absolute calendar positions of the signal days.

    Returns
    -------
    numpy.ndarray
        ``(J, n_states)`` rows on the simplex. Raises if any signal day
        precedes the observables' warm-up - the fold schedule makes that
        impossible, and silence would hide a real alignment bug.
    """
    positions_in_filter = gate.observation_dates.get_indexer(dates[signal_positions])
    if (positions_in_filter < 0).any():
        n_missing = int((positions_in_filter < 0).sum())
        raise ValueError(
            f"{n_missing} signal day(s) precede the HMM observables' warm-up."
        )
    return gate.filtered[positions_in_filter]
