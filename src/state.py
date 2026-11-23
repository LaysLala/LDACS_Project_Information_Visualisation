"""
Shared state of the dashboard: which patients are kept by the filters, and
which are selected.

Two mechanisms (see docs/study_guide.md):
- FILTER  (sidebar): removes patients from every view.
- SELECT  (click / box / lasso in a chart): highlights patients in every view;
  the other patients are drawn in grey and charts can compare
  "selection vs rest". A selection can be turned into a filter
  (Keep only / Exclude).

Only lists of patient ids travel between the browser and the server (dcc.Store),
never data frames: this keeps every callback fast.
"""

from __future__ import annotations

import math

import pandas as pd

from src.data import CATEGORY_ORDER, UNKNOWN, load_data

# --------------------------------------------------------------------------- #
# Filter definitions (one entry per sidebar control)
# --------------------------------------------------------------------------- #
CHECKLIST_FILTERS = {            # control id -> clinical column
    "f-subtype": "pam50_subtype",
    "f-stage": "tumor_stage",
    "f-grade": "neoplasm_histologic_grade",
    "f-vital": "vital_status",
}
RECEPTOR_FILTERS = {             # "All" / "Positive" / "Negative"
    "f-er": "er_status",
    "f-pr": "pr_status",
    "f-her2": "her2_status",
}
TREATMENT_FILTERS = {            # "Any" / "Yes" / "No"
    "f-chemo": "chemotherapy",
    "f-hormone": "hormone_therapy",
    "f-radio": "radio_therapy",
}


def age_bounds() -> tuple[int, int]:
    age = load_data().clinical["age_at_diagnosis"]
    return int(math.floor(age.min())), int(math.ceil(age.max()))


def checklist_options(column: str) -> list[str]:
    """All categories of a column, in display order (Unknown last)."""
    return list(load_data().clinical[column].cat.categories)


def default_filters() -> dict:
    """Sidebar values meaning 'no filter'."""
    values = {cid: checklist_options(col) for cid, col in CHECKLIST_FILTERS.items()}
    values.update({cid: "All" for cid in RECEPTOR_FILTERS})
    values.update({cid: "Any" for cid in TREATMENT_FILTERS})
    values["f-age"] = list(age_bounds())
    return values


EMPTY_MANUAL = {"keep": None, "exclude": []}


def apply_filters(filters: dict, manual: dict | None = None, skip: str | None = None) -> pd.Index:
    """Patient ids kept by the sidebar filters (and by Keep only / Exclude).

    `skip` ignores one control; used to count, next to each option of that
    control, how many patients it would add (scented widget).
    """
    c = load_data().clinical
    mask = pd.Series(True, index=c.index)

    for cid, col in CHECKLIST_FILTERS.items():
        if cid == skip:
            continue
        chosen = filters.get(cid)
        if chosen is not None:
            mask &= c[col].isin(chosen)

    for cid, col in RECEPTOR_FILTERS.items():
        value = filters.get(cid, "All")
        if cid != skip and value != "All":
            mask &= c[col] == value

    for cid, col in TREATMENT_FILTERS.items():
        value = filters.get(cid, "Any")
        if cid != skip and value != "Any":
            mask &= c[col] == value

    if skip != "f-age" and filters.get("f-age"):
        lo, hi = filters["f-age"]
        mask &= c["age_at_diagnosis"].between(lo, hi)

    manual = manual or EMPTY_MANUAL
    if manual.get("keep") is not None:
        mask &= c.index.isin(manual["keep"])
    if manual.get("exclude"):
        mask &= ~c.index.isin(manual["exclude"])

    return c.index[mask]


def visible_selection(selected: dict | None, filtered_ids) -> list:
    """Selected patients that are still kept by the filters."""
    if not selected or not selected.get("ids"):
        return []
    kept = set(filtered_ids)
    return [i for i in selected["ids"] if i in kept]


# Sidebar control matching each clinical column: a selection made on one of these columns
# ("subtype = Basal", "ER = Negative", "age 40-50") can be turned into that control's value,
# so "Keep only" / "Exclude" visibly update the sidebar instead of a hidden list of patients.
CONTROL_FOR = {
    **{col: cid for cid, col in CHECKLIST_FILTERS.items()},
    **{col: cid for cid, col in RECEPTOR_FILTERS.items()},
    **{col: cid for cid, col in TREATMENT_FILTERS.items()},
    "age_at_diagnosis": "f-age",
}
OPPOSITE = {"Positive": "Negative", "Negative": "Positive", "Yes": "No", "No": "Yes"}


def make_selection(ids, source: str, column: str | None = None, value=None) -> dict:
    """Store payload for a new selection (ids as plain ints for JSON).

    `column` / `value` describe the selection when it is "all filtered patients with
    column == value" (or, for age, value = [low, high]); only kept for columns that have a
    sidebar control."""
    payload = {"ids": [int(i) for i in ids], "source": source}
    if column in CONTROL_FOR:
        payload["filter"] = {"column": column, "value": value}
    return payload


def selection_as_filter(spec: dict, keep: bool, current: dict) -> dict | None:
    """New sidebar values that keep only / exclude a selection described by `spec`
    ({column, value}); None when no sidebar control can express it."""
    cid = CONTROL_FOR.get(spec.get("column"))
    value = spec.get("value")
    if cid is None:
        return None
    if cid in CHECKLIST_FILTERS:
        chosen = current.get(cid) or []
        return {cid: [value] if keep else [v for v in chosen if v != value]}
    if cid in RECEPTOR_FILTERS or cid in TREATMENT_FILTERS:
        if value not in OPPOSITE:
            return None
        return {cid: value if keep else OPPOSITE[value]}
    if cid == "f-age" and keep:                       # a range can be kept, not excluded, with one slider
        lo, hi = age_bounds()
        return {cid: [max(lo, math.floor(value[0])), min(hi, math.ceil(value[1]))]}
    return None


def categorical_columns(max_categories: int = 8) -> list[str]:
    """Clinical categorical columns small enough to split or colour a chart by."""
    from src.data import CATEGORICAL_CLINICAL

    c = load_data().clinical
    out = []
    for col in CATEGORICAL_CLINICAL:
        cats = [x for x in c[col].cat.categories if x != UNKNOWN]
        if 2 <= len(cats) <= max_categories:
            out.append(col)
    return out


def ordered_categories(column: str, values: pd.Series) -> list[str]:
    """Categories present in `values`, in the column's display order."""
    order = CATEGORY_ORDER.get(column) or checklist_options(column)
    present = set(values.astype(str))
    ordered = [c for c in order if c in present]
    rest = sorted(present - set(ordered) - {UNKNOWN})
    tail = [UNKNOWN] if UNKNOWN in present else []
    return ordered + rest + tail
