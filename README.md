# METABRIC Explorer

An interactive visual analytics dashboard for exploring the METABRIC breast
cancer cohort: 1,904 patients described by 31 clinical attributes, mRNA
expression z-scores for 489 genes and mutation status for 173 genes.

It is aimed at a researcher who wants to understand how the molecular subtypes
of breast cancer relate to clinical features, mutations and survival.

Developed for the Information Visualization course project (LDACS2210, UCLouvain).

**Authors:** Lays Zandomingos and Renan Toledo
**Year:** 2026

---

## 1. Dataset

The dataset is **not included** in the submission.

1. Download `METABRIC_RNA_Mutation.csv` from Kaggle:
   <https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric>
2. Put it, without renaming it, in the `data` folder at the root of the project:

```
<project root>/
├── app.py
├── data/
│   └── METABRIC_RNA_Mutation.csv   <-- here
└── ...
```

If the file is missing, the application stops with a message saying where to
put it.

---

## 2. Installation

Requires **Python 3.10 or newer**.

> **Windows:** if `python` opens the Microsoft Store, Python is not on the PATH; use the
> launcher `py` instead of `python` in the commands below (e.g. `py -m venv .venv`).

```bash
# 1. create and activate a virtual environment (from the project root)
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. install the dependencies
pip install -r requirements.txt
```

To check that the dataset is found and read correctly:

```bash
python -m src.data
```

---

## 3. Running the dashboard

```bash
python app.py
```

Then open <http://127.0.0.1:8050> in a web browser. A short guide opens on the first visit
(it can be reopened with the **Help** button).

---

## 4. Project structure

```
├── app.py                  entry point of the dashboard (layout shell)
├── requirements.txt        Python dependencies
├── data/                   the dataset goes here (not included)
├── pages/                  one module per dashboard page (Dash Pages)
│   ├── overview.py
│   ├── relationships.py
│   └── clusters.py
├── assets/style.css        page layout and component styles
├── src/
│   ├── data.py             loading and cleaning (cleaning decisions documented inside)
│   ├── state.py            filters and selection (shared state)
│   ├── callbacks.py        app-wide callbacks (filters, selection, header, help)
│   ├── components.py       header, filter sidebar, selection bar, cards
│   ├── figures.py          figure builders (Overview page)
│   ├── figures_relationships.py   figure builders (Relationships & genomics page)
│   ├── figures_clusters.py figure builders (Clusters & projections page)
│   ├── compute.py          cached projections and clusterings (heavy computations)
│   ├── variables.py        clinical / gene / mutation variables shared by the charts
│   ├── stats.py            Kaplan-Meier, log-rank test, mutual information
│   ├── algorithms.py       dimensionality reduction, clustering, quality measures
│   └── theme.py            shared colours, fonts and Plotly template
├── notebooks/
│   └── Exploratory.ipynb   exploratory data analysis
└── report/                 project report
```

All file paths in the code are relative to the project root.

---

## 5. Exploratory notebook (optional)

`notebooks/Exploratory.ipynb` contains the exploratory analysis behind the
design of the dashboard. Besides `requirements.txt`, it needs:

```bash
pip install jupyter matplotlib xgboost shap
```

Open it from the `notebooks/` folder and run all cells (about 2 minutes).
