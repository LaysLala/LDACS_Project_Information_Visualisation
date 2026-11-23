"""
Data access layer for the METABRIC dashboard.

Every page of the app reads the dataset through `load_data()`, so all views see
exactly the same cleaned data. The cleaning decisions are listed here because
the report has to justify them.

Cleaning decisions
------------------
1. Three blocks. The CSV mixes three kinds of columns, which we split:
     - clinical  : 31 patient attributes (+ a few derived columns, see 5.)
     - mrna      : 489 mRNA expression z-scores (no missing values)
     - mutations : 173 genes, 1 = the patient has >= 1 mutation in that gene
2. Mutation columns store mutation codes as text ('H1047R', 'Q546H G1049R', ...)
   and '0' when the gene is not mutated. One-hot encoding them would create
   ~8,000 almost-empty columns that drown the real signal, so we binarise them
   (mutated yes/no) and keep the raw codes separately for tooltips.
3. Missing values are NOT imputed in the clinical block. Missing categories
   become an explicit "Unknown" category (the user sees how much is missing);
   missing numbers stay NaN and are left out chart by chart. Clustering and
   dimensionality reduction run on the mRNA block, which is complete, so they
   need no imputation at all.
   One exception, backed by the data: `mutation_count` is missing for exactly
   the 45 patients with no mutated gene in the panel (and tracks the number of
   mutated genes with r = 0.97 elsewhere), so missing means 0, not unknown.
4. Encoding fixes: the typo 'Positve' -> 'Positive'; 0/1 treatment flags ->
   'No'/'Yes'; ordinal variables (grade, stage, cellularity, ...) get an
   explicit order so every chart and filter lists them the same way.
5. Survival. `overall_survival` is 1 when the patient is alive. We derive:
     - os_event     : 1 = died of any cause            (overall survival)
     - dss_event    : 1 = died of breast cancer, 0 = alive or died of another
                      cause, NaN = cause unknown       (disease-specific survival)
     - vital_status : Living / Died of disease / Died of other causes
6. Near-constant or redundant columns stay in the table but are not offered in
   the UI: cancer_type (1903 of 1904 are 'Breast Cancer'), oncotree_code (same
   information as cancer_type_detailed), cohort (sampling batch, no clinical
   meaning).
7. Gene expression is used as published. The exploratory notebook found a
   technical difference between the 5 sampling cohorts (notebooks/
   Exploratory.ipynb, section 7); correcting it was judged too hard for a
   dashboard user to interpret, so the dashboard keeps the published values.

Run `python -m src.data` from the project root to print a summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

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
    "pam50_subtype": "PAM50 subtype",
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
