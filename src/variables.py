"""
Variables the user can pick in a chart (pair view, colour-by, ...):

    clin:<column>   a clinical variable (numeric or categorical)
    gene:<gene>     mRNA expression of a gene (numeric, z-score)
    mut:<gene>      mutation status of a gene (categorical: Mutated / Not mutated)

One place decides the type, the label and the display order of every
variable, so all pages treat them identically.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data import CATEGORICAL_CLINICAL, NUMERIC_CLINICAL, gene_label, label, load_data, mrna_matrix
from src.state import ordered_categories

MUTATED, NOT_MUTATED = "Mutated", "Not mutated"


@dataclass(frozen=True)
class Variable:
    key: str
    kind: str      # "num" or "cat"
    label: str     # axis / legend title, with units
    short: str     # compact name for titles


def describe(key: str) -> Variable:
    source, name = key.split(":", 1)
    if source == "clin":
        kind = "num" if name in NUMERIC_CLINICAL else "cat"
        return Variable(key, kind, label(name), label(name).split(" (")[0])
    if source == "gene":
        return Variable(key, "num", gene_label(name), name.upper())
    if source == "mut":
        return Variable(key, "cat", f"{name.upper()} mutation status", f"{name.upper()} mutation")
    raise ValueError(f"unknown variable {key!r}")


def values(key: str, ids, corrected: bool = False) -> pd.Series:
    """Values of a variable for the given patients (index = patient id)."""
    source, name = key.split(":", 1)
    d = load_data()
    if source == "clin":
        s = d.clinical.loc[ids, name]
        return s.astype(str) if describe(key).kind == "cat" else s
    if source == "gene":
        return mrna_matrix(corrected).loc[ids, name]
    return d.mutations.loc[ids, name].map({1: MUTATED, 0: NOT_MUTATED})


def categories(key: str, vals: pd.Series) -> list[str]:
    """Display order of a categorical variable's values."""
    source, name = key.split(":", 1)
    if source == "mut":
        return [c for c in (MUTATED, NOT_MUTATED) if c in set(vals)]
    return ordered_categories(name, vals)


def options() -> list[dict]:
    """Dropdown options: clinical first, then all genes, then all mutations (searchable)."""
    d = load_data()
    opts = [{"label": f"Clinical · {label(c)}", "value": f"clin:{c}"} for c in NUMERIC_CLINICAL + CATEGORICAL_CLINICAL]
    opts += [{"label": f"Gene · {g.upper()} expression", "value": f"gene:{g}"} for g in sorted(d.genes)]
    opts += [{"label": f"Mutation · {g.upper()}", "value": f"mut:{g}"} for g in sorted(d.mutated_genes)]
    return opts
