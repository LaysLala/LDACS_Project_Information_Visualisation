# METABRIC Explorer

An interactive visual analytics dashboard for exploring the METABRIC breast
cancer cohort (1,904 patients, 489 genes, 173 mutation panels).

Developed for the Information Visualization course project.

**Author:** Lays Zandomingos and Renan Toledo
**Course:** Information Visualization  
**Year:** 2026

---

## 1. Dataset

The dataset is **not included** in this repository.

Download `METABRIC_RNA_Mutation.csv` from Kaggle:
<https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric>

Place the file here, without renaming it:

```
metabric-explorer/data/METABRIC_RNA_Mutation.csv
```

The application will not start without it, and will tell you so explicitly.

---

## 2. Installation

Requires Python 3.9 or newer.

```bash
# 1. create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Mac / Linux
source .venv/bin/activate

# 2. install the dependencies
pip install -r requirements.txt
