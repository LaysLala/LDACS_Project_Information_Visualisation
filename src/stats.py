"""
Statistics used by the dashboard and the exploratory notebook.

- Survival share over time: for each month t, the share of patients still alive
  among the patients whose status at t is known. A patient alive at the end of
  their follow-up only tells us they survived *at least* that long (censoring),
  so patients whose follow-up ended before t, still alive, are left out at t.
  A simple proportion per month: accurate over the first years, when almost
  every patient's status is known; that is why the dashboard shows the first
  5 years only (later, patients recruited later in the study would be missing
  from the share and the curve would drop too fast).
- Normalized mutual information: strength of association between two
  categorical variables (0 = independent, 1 = one determines the other).

References
----------
Strehl & Ghosh (2002), Cluster ensembles (normalized mutual information). JMLR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

HORIZON = 60          # months shown: the first 5 years
MIN_KNOWN = 10        # a curve stops when fewer patients have a known status


def survival_share(time, event, horizon: int = HORIZON) -> pd.DataFrame:
    """Share of patients alive at each month 0..horizon, among patients whose status is known.

    At month t:
        alive = patients followed for more than t months
        dead  = patients whose event (death) happened at or before t
        share = alive / (alive + dead)
    Rows where fewer than MIN_KNOWN patients have a known status are dropped.
    Inputs are matched by position; rows with a missing time or event are ignored.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=float)
    keep = ~(np.isnan(t) | np.isnan(e))
    t, e = t[keep], e[keep]
    months = np.arange(0, horizon + 1)
    alive = (t[None, :] > months[:, None]).sum(axis=1)
    dead = ((e[None, :] == 1) & (t[None, :] <= months[:, None])).sum(axis=1)
    known = alive + dead
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(known > 0, alive / known, np.nan)
    out = pd.DataFrame({"month": months, "alive": alive, "dead": dead, "known": known, "survival": share})
    return out[out["known"] >= MIN_KNOWN]


def survival_at(curve: pd.DataFrame, month: int) -> float:
    """Value of a survival_share curve at a given month (NaN if not available)."""
    row = curve.loc[curve["month"] == month, "survival"]
    return float(row.iloc[0]) if len(row) else float("nan")


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
