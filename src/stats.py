"""
Statistics used by the dashboard and the exploratory notebook.

- Kaplan-Meier estimator of the survival function (with Greenwood confidence
  band) and the k-group log-rank test. Survival data is "time-to-event" with
  censoring: a patient alive at the end of follow-up only tells us she survived
  *at least* that long, so a simple "% deceased" bar would be misleading.
- Normalized mutual information: strength of association between two
  categorical variables (0 = independent, 1 = one determines the other).

References
----------
Kaplan & Meier (1958), Nonparametric estimation from incomplete observations.
Mantel (1966), Evaluation of survival data and two new rank order statistics.
Strehl & Ghosh (2002), Cluster ensembles (normalized mutual information). JMLR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import normalized_mutual_info_score


def kaplan_meier(time, event, confidence: float = 0.95) -> pd.DataFrame:
    """Kaplan-Meier survival curve.

    Parameters
    ----------
    time  : follow-up time per patient (months)
    event : 1 if the event (death) happened at `time`, 0 if censored.
            Rows where time or event is NaN are ignored.

    Returns
    -------
    DataFrame with one row per distinct time (plus t = 0):
    time, at_risk, events, censored, survival, ci_low, ci_high.
    Draw it as a step function (`line_shape="hv"`).
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=float)
    keep = ~(np.isnan(t) | np.isnan(e))
    t, e = t[keep], e[keep].astype(int)
    n = len(t)
    if n == 0:
        return pd.DataFrame(
            columns=["time", "at_risk", "events", "censored", "survival", "ci_low", "ci_high"]
        )

    times = np.unique(t)
    t_sorted = np.sort(t)
    at_risk = n - np.searchsorted(t_sorted, times, side="left")
    idx = np.searchsorted(times, t)
    events = np.bincount(idx, weights=e, minlength=len(times))
    censored = np.bincount(idx, weights=1 - e, minlength=len(times))

    hazard = events / at_risk
    survival = np.cumprod(1.0 - hazard)

    # Greenwood variance on the log(-log S) scale -> band stays inside [0, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        gw = np.cumsum(events / (at_risk * (at_risk - events)))
        z = sps.norm.ppf(0.5 + confidence / 2)
        log_s = np.log(survival)
        se = np.sqrt(gw) / np.abs(log_s)
        ci_low = np.exp(-np.exp(np.log(-log_s) + z * se))
        ci_high = np.exp(-np.exp(np.log(-log_s) - z * se))
    ci_low = np.where(np.isfinite(ci_low), ci_low, survival)
    ci_high = np.where(np.isfinite(ci_high), ci_high, survival)

    curve = pd.DataFrame(
        {
            "time": times,
            "at_risk": at_risk,
            "events": events.astype(int),
            "censored": censored.astype(int),
            "survival": survival,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    )
    start = pd.DataFrame(
        {"time": [0.0], "at_risk": [n], "events": [0], "censored": [0],
         "survival": [1.0], "ci_low": [1.0], "ci_high": [1.0]}
    )
    if times[0] > 0:
        curve = pd.concat([start, curve], ignore_index=True)
    return curve


def median_survival(curve: pd.DataFrame) -> float:
    """First time at which the survival curve drops to 50% or below (NaN if never)."""
    below = curve.loc[curve["survival"] <= 0.5, "time"]
    return float(below.iloc[0]) if len(below) else float("nan")


def survival_at(curve: pd.DataFrame, t: float) -> float:
    """Survival probability at time t (e.g. 60 months = 5-year survival)."""
    before = curve.loc[curve["time"] <= t, "survival"]
    return float(before.iloc[-1]) if len(before) else 1.0


def logrank_test(time, event, groups) -> tuple[float, int, float]:
    """k-group log-rank test: are the survival curves of the groups identical?

    Returns (chi2 statistic, degrees of freedom, p-value).
    Inputs are matched by position (not by pandas index).
    """
    df = pd.DataFrame(
        {"t": np.asarray(time, dtype=float), "e": np.asarray(event, dtype=float),
         "g": np.asarray(groups, dtype=object)}
    ).dropna()
    df["e"] = df["e"].astype(int)
    labels = list(pd.unique(df["g"]))
    k = len(labels)
    if k < 2:
        return float("nan"), 0, float("nan")

    times = np.sort(df.loc[df["e"] == 1, "t"].unique())
    n_at_risk = np.empty((k, len(times)))
    d_events = np.empty((k, len(times)))
    for gi, lab in enumerate(labels):
        sub = df[df["g"] == lab]
        ts = np.sort(sub["t"].to_numpy())
        n_at_risk[gi] = len(ts) - np.searchsorted(ts, times, side="left")
        ev = sub.loc[sub["e"] == 1, "t"].to_numpy()
        d_events[gi] = np.bincount(np.searchsorted(times, ev), minlength=len(times))

    n_tot = n_at_risk.sum(axis=0)
    d_tot = d_events.sum(axis=0)
    observed = d_events.sum(axis=1)
    expected = (d_tot * n_at_risk / n_tot).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(n_tot > 1, d_tot * (n_tot - d_tot) / (n_tot - 1), 0.0)
        frac = n_at_risk / n_tot
    var = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            delta = 1.0 if a == b else 0.0
            var[a, b] = np.sum(factor * frac[a] * (delta - frac[b]))

    o_minus_e = (observed - expected)[:-1]
    stat = float(o_minus_e @ np.linalg.pinv(var[:-1, :-1]) @ o_minus_e)
    dof = k - 1
    return stat, dof, float(sps.chi2.sf(stat, dof))


def mutual_information(x, y) -> float:
    """Normalized mutual information between two categorical variables.

    Mutual information (TP1) measures how much knowing one variable tells
    about the other. Normalising by the entropies (arithmetic mean) maps it to
    [0, 1]: 0 = independent, 1 = one determines the other, so pairs with
    different numbers of categories can be compared on one colour scale.
    Inputs are matched by position (not by pandas index)."""
    return float(normalized_mutual_info_score(np.asarray(x).astype(str),
                                              np.asarray(y).astype(str)))


def mutual_information_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Symmetric matrix of normalized mutual information for categorical columns."""
    m = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            m.loc[a, b] = m.loc[b, a] = mutual_information(df[a], df[b])
    return m
