"""
Data layer of the METABRIC dashboard: which patients, and which values.

Sections
1. Loading and cleaning: `load_data()`; every cleaning decision is listed there.
2. Filters and selection: the shared state of the app (sidebar filters, selected patients).
3. Variables: one registry for clinical attributes, genes and mutations
   (`clin:` / `gene:` / `mut:` keys), so every chart treats a variable the same way.

Run `python -m src.data` to check that the dataset is found and read correctly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd


# =============================================================================
# 1. Loading and cleaning
# =============================================================================
# Data access layer for the METABRIC dashboard.
#
# Every page of the app reads the dataset through `load_data()`, so all views see
# exactly the same cleaned data. The cleaning decisions are listed here because
# the report has to justify them.
#
# Cleaning decisions
# ------------------
# 1. Three blocks. The CSV mixes three kinds of columns, which we split:
#      - clinical  : 31 patient attributes (+ a few derived columns, see 5.)
#      - mrna      : 489 mRNA expression z-scores (no missing values)
#      - mutations : 173 genes, 1 = the patient has >= 1 mutation in that gene
# 2. Mutation columns store mutation codes as text ('H1047R', 'Q546H G1049R', ...)
#    and '0' when the gene is not mutated. One-hot encoding them would create
#    ~8,000 almost-empty columns that drown the real signal, so we binarise them
#    (mutated yes/no) and keep the raw codes separately for tooltips.
# 3. Missing values are NOT imputed in the clinical block. Missing categories
#    become an explicit "Unknown" category (the user sees how much is missing);
#    missing numbers stay NaN and are left out chart by chart. Clustering and
#    dimensionality reduction run on the mRNA block, which is complete, so they
#    need no imputation at all.
#    One exception, backed by the data: `mutation_count` is missing for exactly
#    the 45 patients with no mutated gene in the panel (and tracks the number of
#    mutated genes with r = 0.97 elsewhere), so missing means 0, not unknown.
# 4. Encoding fixes: the typo 'Positve' -> 'Positive'; 0/1 treatment flags ->
#    'No'/'Yes'; ordinal variables (grade, stage, cellularity, ...) get an
#    explicit order so every chart and filter lists them the same way.
# 5. Survival. `overall_survival` is 1 when the patient is alive. We derive:
#      - os_event     : 1 = died of any cause            (overall survival)
#      - dss_event    : 1 = died of breast cancer, 0 = alive or died of another
#                       cause, NaN = cause unknown       (disease-specific survival)
#      - vital_status : Living / Died of disease / Died of other causes
# 6. Near-constant or redundant columns stay in the table but are not offered in
#    the UI: cancer_type (1903 of 1904 are 'Breast Cancer'), oncotree_code (same
#    information as cancer_type_detailed), cohort (sampling batch, no clinical
#    meaning).
# 7. Gene expression is used as published. The exploratory notebook found a
#    technical difference between the 5 sampling cohorts (notebooks/
#    Exploratory.ipynb, section 7); correcting it was judged too hard for a
#    dashboard user to interpret, so the dashboard keeps the published values.
#
# Run `python -m src.data` from the project root to print a summary.

# Locations (always relative to the project, never absolute)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "METABRIC_RNA_Mutation.csv"
DATASET_URL = (
    "https://www.kaggle.com/datasets/raghadalharbi/"
    "breast-cancer-gene-expression-profiles-metabric"
)

FIRST_GENE_COLUMN = "brca1"  # first mRNA column in the CSV
MUTATION_SUFFIX = "_mut"
UNKNOWN = "Unknown"

# Column vocabulary
RENAME = {
    "pam50_+_claudin-low_subtype": "pam50_subtype",
    "3-gene_classifier_subtype": "three_gene_subtype",
}

# Numeric clinical variables offered in the UI.
NUMERIC_CLINICAL = [
    "age_at_diagnosis",
    "tumor_size",
    "lymph_nodes_examined_positive",
    "nottingham_prognostic_index",
    "mutation_count",
    "overall_survival_months",
]

# Categorical clinical variables offered in the UI, roughly by importance.
CATEGORICAL_CLINICAL = [
    "pam50_subtype",
    "vital_status",
    "er_status",
    "pr_status",
    "her2_status",
    "three_gene_subtype",
    "integrative_cluster",
    "neoplasm_histologic_grade",
    "tumor_stage",
    "cellularity",
    "tumor_other_histologic_subtype",
    "cancer_type_detailed",
    "inferred_menopausal_state",
    "type_of_breast_surgery",
    "chemotherapy",
    "hormone_therapy",
    "radio_therapy",
    "er_status_measured_by_ihc",
    "her2_status_measured_by_snp6",
    "primary_tumor_laterality",
]

# Explicit display order for ordinal / well-known categories. Categories not
# listed here are ordered by frequency (most common first). "Unknown" is
# always appended last.
CATEGORY_ORDER: dict[str, list[str]] = {
    "pam50_subtype": ["LumA", "LumB", "Her2", "Basal", "claudin-low", "Normal", "NC"],
    "vital_status": ["Living", "Died of disease", "Died of other causes"],
    "integrative_cluster": ["1", "2", "3", "4ER+", "4ER-", "5", "6", "7", "8", "9", "10"],
    "neoplasm_histologic_grade": ["1", "2", "3"],
    "tumor_stage": ["0", "1", "2", "3", "4"],
    "cellularity": ["Low", "Moderate", "High"],
    "three_gene_subtype": [
        "ER+/HER2- Low Prolif",
        "ER+/HER2- High Prolif",
        "ER-/HER2-",
        "HER2+",
    ],
    "her2_status_measured_by_snp6": ["LOSS", "NEUTRAL", "GAIN", "UNDEF"],
    "inferred_menopausal_state": ["Pre", "Post"],
    "er_status": ["Positive", "Negative"],
    "pr_status": ["Positive", "Negative"],
    "her2_status": ["Positive", "Negative"],
    "er_status_measured_by_ihc": ["Positive", "Negative"],
    "chemotherapy": ["Yes", "No"],
    "hormone_therapy": ["Yes", "No"],
    "radio_therapy": ["Yes", "No"],
}

# Human-readable labels, with units, for axes, legends and filters.
LABELS: dict[str, str] = {
    "age_at_diagnosis": "Age at diagnosis (years)",
    "tumor_size": "Tumor size (mm)",
    "lymph_nodes_examined_positive": "Positive lymph nodes (count)",
    "nottingham_prognostic_index": "Nottingham Prognostic Index",
    "mutation_count": "Mutation count (sequenced panel)",
    "n_mutated_genes": "Mutated genes (of 173 in panel)",
    "overall_survival_months": "Follow-up time (months)",
    "pam50_subtype": "Molecular subtype (PAM50 + claudin-low)",
    "vital_status": "Vital status",
    "er_status": "ER status",
    "pr_status": "PR status",
    "her2_status": "HER2 status",
    "three_gene_subtype": "3-gene classifier",
    "integrative_cluster": "Integrative cluster",
    "neoplasm_histologic_grade": "Histologic grade",
    "tumor_stage": "Tumor stage",
    "cellularity": "Cellularity",
    "tumor_other_histologic_subtype": "Histologic subtype",
    "cancer_type_detailed": "Cancer type (detailed)",
    "inferred_menopausal_state": "Menopausal state",
    "type_of_breast_surgery": "Surgery",
    "chemotherapy": "Chemotherapy",
    "hormone_therapy": "Hormone therapy",
    "radio_therapy": "Radiotherapy",
    "er_status_measured_by_ihc": "ER status (IHC)",
    "her2_status_measured_by_snp6": "HER2 copy number (SNP6)",
    "primary_tumor_laterality": "Tumor laterality",
    "os_event": "Death (any cause)",
    "dss_event": "Death from breast cancer",
}


# Public API
class DatasetMissingError(FileNotFoundError):
    """Raised when the CSV is not where the README says it should be."""


@dataclass(frozen=True)
class Metabric:
    """The cleaned dataset. All frames share the same index (patient_id).

    Treat the frames as read-only: they are cached and shared by every view.
    """

    clinical: pd.DataFrame        # cleaned clinical + derived columns
    mrna: pd.DataFrame            # 489 mRNA z-scores, float
    mutations: pd.DataFrame       # 173 genes, int8 0/1 (column = gene symbol)
    mutation_codes: pd.DataFrame  # raw mutation codes, "" when not mutated

    @property
    def n_patients(self) -> int:
        return len(self.clinical)

    @property
    def genes(self) -> list[str]:
        return list(self.mrna.columns)

    @property
    def mutated_genes(self) -> list[str]:
        return list(self.mutations.columns)


@lru_cache(maxsize=1)
def load_data(path: Path | str = DATA_PATH) -> Metabric:
    """Read and clean the dataset once; later calls return the cached object."""
    raw = _read_raw(Path(path))
    clinical_cols, gene_cols, mutation_cols = _split_blocks(raw)

    raw = raw.set_index("patient_id")
    clinical_cols = [c for c in clinical_cols if c != "patient_id"]

    mrna = raw[gene_cols].astype(float)

    codes = raw[mutation_cols].astype(str).rename(columns=_strip_mut_suffix)
    mutated = codes != "0"
    mutations = mutated.astype("int8")
    mutation_codes = codes.where(mutated, "")

    clinical = _clean_clinical(raw[clinical_cols])
    clinical["n_mutated_genes"] = mutations.sum(axis=1).astype(int)
    # 3. (exception) missing mutation_count == no mutation found in the panel
    no_mutation = clinical["n_mutated_genes"] == 0
    clinical.loc[no_mutation & clinical["mutation_count"].isna(), "mutation_count"] = 0

    return Metabric(
        clinical=clinical,
        mrna=mrna,
        mutations=mutations,
        mutation_codes=mutation_codes,
    )


def label(column: str) -> str:
    """Display name for any column: clinical, gene expression or mutation."""
    if column in LABELS:
        return LABELS[column]
    if column.endswith(MUTATION_SUFFIX):
        return f"{column[: -len(MUTATION_SUFFIX)].upper()} mutation"
    return column.replace("_", " ").capitalize()


def gene_label(gene: str) -> str:
    """Display name for an mRNA column, e.g. 'brca1' -> 'BRCA1 expression (z-score)'."""
    return f"{gene.upper()} expression (z-score)"


def category_order(clinical: pd.DataFrame, column: str) -> list[str]:
    """Categories of a clinical column in display order (for Plotly category_orders)."""
    return list(clinical[column].cat.categories)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _read_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        try:
            shown = path.relative_to(PROJECT_ROOT)
        except ValueError:
            shown = path
        raise DatasetMissingError(
            f"Dataset not found: {shown}\n"
            f"Download METABRIC_RNA_Mutation.csv from\n  {DATASET_URL}\n"
            "and put it in the 'data' folder of the project (see README.md)."
        )
    return pd.read_csv(path, low_memory=False)


def _split_blocks(raw: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Split column names into (clinical, mRNA, mutation) using the CSV layout."""
    cols = list(raw.columns)
    if FIRST_GENE_COLUMN not in cols:
        raise ValueError(f"Unexpected CSV layout: no '{FIRST_GENE_COLUMN}' column.")
    first_gene = cols.index(FIRST_GENE_COLUMN)
    first_mut = next(
        (i for i, c in enumerate(cols) if c.endswith(MUTATION_SUFFIX)), None
    )
    if first_mut is None or first_mut < first_gene:
        raise ValueError("Unexpected CSV layout: mutation columns not found after genes.")
    return cols[:first_gene], cols[first_gene:first_mut], cols[first_mut:]


def _strip_mut_suffix(column: str) -> str:
    return column[: -len(MUTATION_SUFFIX)] if column.endswith(MUTATION_SUFFIX) else column


def _clean_clinical(df: pd.DataFrame) -> pd.DataFrame:
    c = df.rename(columns=RENAME).copy()

    # 4. encoding fixes
    c["er_status_measured_by_ihc"] = c["er_status_measured_by_ihc"].replace(
        {"Positve": "Positive"}
    )
    for col in ("chemotherapy", "hormone_therapy", "radio_therapy"):
        c[col] = c[col].map({0: "No", 1: "Yes"})
    for col in ("neoplasm_histologic_grade", "tumor_stage"):
        c[col] = c[col].map(lambda v: str(int(v)), na_action="ignore")
    c["type_of_breast_surgery"] = c["type_of_breast_surgery"].str.capitalize()

    # 5. survival
    c["os_event"] = (c["overall_survival"] == 0).astype(int)
    cause = c["death_from_cancer"]
    c["dss_event"] = cause.map(
        {"Died of Disease": 1.0, "Died of Other Causes": 0.0, "Living": 0.0}
    )
    c["vital_status"] = cause.map(
        {
            "Living": "Living",
            "Died of Disease": "Died of disease",
            "Died of Other Causes": "Died of other causes",
        }
    )

    # 3./4. categorical columns: explicit "Unknown" + fixed order
    for col in CATEGORICAL_CLINICAL:
        c[col] = _as_ordered_category(c[col], CATEGORY_ORDER.get(col))

    return c


def _as_ordered_category(series: pd.Series, order: list[str] | None) -> pd.Series:
    values = series.astype("string").fillna(UNKNOWN)
    present = set(values.unique())
    if order is None:
        order = [v for v in values.value_counts().index if v != UNKNOWN]
    else:
        # keep the declared order, but never drop an unexpected value
        order = [v for v in order if v in present] + sorted(
            present - set(order) - {UNKNOWN}
        )
    if UNKNOWN in present:
        order = order + [UNKNOWN]
    return pd.Categorical(values, categories=order, ordered=False)


# Quick check:  python -m src.data


# =============================================================================
# 2. Filters and selection (shared state of the app)
# =============================================================================
# Shared state of the dashboard: which patients are kept by the filters, and
# which are selected.
#
# Two mechanisms (see docs/study_guide.md):
# - FILTER  (sidebar): removes patients from every view.
# - SELECT  (click / box / lasso in a chart): highlights patients in every view;
#   the other patients are drawn in grey and charts can compare
#   "selection vs rest". A selection can be turned into a filter
#   (Keep only / Exclude).
#
# Only lists of patient ids travel between the browser and the server (dcc.Store),
# never data frames: this keeps every callback fast.

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


# =============================================================================
# 3. Variables: clinical attributes, genes and mutations
# =============================================================================
# Variables the user can pick in a chart (pair view, colour-by, ...):
#
#     clin:<column>   a clinical variable (numeric or categorical)
#     gene:<gene>     mRNA expression of a gene (numeric, z-score)
#     mut:<gene>      mutation status of a gene (categorical: Mutated / Not mutated)
#
# One place decides the type, the label and the display order of every
# variable, so all pages treat them identically.

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


def values(key: str, ids) -> pd.Series:
    """Values of a variable for the given patients (index = patient id)."""
    source, name = key.split(":", 1)
    d = load_data()
    if source == "clin":
        s = d.clinical.loc[ids, name]
        return s.astype(str) if describe(key).kind == "cat" else s
    if source == "gene":
        return d.mrna.loc[ids, name]
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


if __name__ == "__main__":
    d = load_data()
    print(f"patients            : {d.n_patients}")
    print(f"clinical columns    : {d.clinical.shape[1]}")
    print(f"mRNA genes          : {len(d.genes)}   (missing values: {int(d.mrna.isna().sum().sum())})")
    print(f"mutation genes      : {len(d.mutated_genes)}")
    print(f"patients w/ mutation: {int((d.mutations.sum(axis=1) > 0).sum())}")
    print()
    print("Unknown per categorical column:")
    for col in CATEGORICAL_CLINICAL:
        n = int((d.clinical[col] == UNKNOWN).sum())
        if n:
            print(f"  {label(col):<32} {n:>4}  ({n / d.n_patients:.1%})")
    print()
    print("Missing per numeric column:")
    for col in NUMERIC_CLINICAL:
        n = int(d.clinical[col].isna().sum())
        if n:
            print(f"  {label(col):<32} {n:>4}  ({n / d.n_patients:.1%})")
    print()
    print("PAM50 subtype order:", category_order(d.clinical, "pam50_subtype"))
    print("Vital status counts:", d.clinical["vital_status"].value_counts(sort=False).to_dict())
