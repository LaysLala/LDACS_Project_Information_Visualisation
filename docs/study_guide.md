# METABRIC Explorer: study guide

*For Renan and Lays. Last updated: end of Phase 3 (2026-11-23). Update it at the end of every phase.*

This file explains **what we decided, why, how it connects to the course, and what a teacher is likely to ask**.
Read it before the TA meeting and again before the oral exam. Every number quoted here comes from
`notebooks/Exploratory.ipynb` or the code in `src/`; if you change the analysis, update the numbers here.

---

## Contents

1. [The project in one page](#1-the-project-in-one-page)
2. [Decision log by phase](#2-decision-log-by-phase)
3. [Course mapping: where each technique comes from](#3-course-mapping)
4. [Teacher questions and answers](#4-teacher-questions-and-answers)
5. [Key numbers cheat sheet](#5-key-numbers-cheat-sheet)
6. [Known limitations (say them before they are asked)](#6-known-limitations)
7. [Glossary](#7-glossary)

---

## 1. The project in one page

| | |
|---|---|
| **Course** | LDACS2210 Information Visualization (UCLouvain) |
| **Deliverables** | Week 10: EDA + UI sketch (one-on-one with a TA). Final: Dash app + report (max 7 pages), zip on Moodle by **Fri 11 Dec 2026, 23:55**. Oral demo at the exam. |
| **Grading** | `course_material/Evaluation_grid_LDATA2010.pdf`: data representation, interaction (linked views, tooltips, speed), best practices (axes, density, consistency), usability (navigation, help, feedback, layout without scrolling, accessibility), creativity, report. |
| **Dataset** | METABRIC: 1,904 breast-cancer patients; 31 clinical columns, mRNA z-scores for **489** genes, mutation calls for **173** genes (the brief says 331/175; the file says 489/173). |
| **Target user** | A researcher exploring how the molecular subtypes of breast cancer relate to clinical features, mutations and survival. |
| **Stack** | Python, Dash + Plotly (required by the course: Dash practical session), scikit-learn, SciPy, umap-learn. |

**Dashboard (3 pages)**, see the UI sketch canvas *METABRIC Explorer UI sketch v2*:

| Page | Cards | User questions answered |
|---|---|---|
| **1. Overview** | KPI tiles · patients per subtype · Kaplan–Meier survival · distribution of a variable · categorical breakdown | Q1 what does my (filtered) cohort look like? Q3 does survival differ between groups? |
| **2. Relationships & genomics** | association matrix (mutual information / Spearman) · adaptive pair view · mutation frequency · cluster heatmap | Q2 how are variables linked? Q4 which genes / mutations differ in a group? |
| **3. Clusters & projections** | settings + Run · projection scatter · quality (choose-k, R_NX, dendrogram) · clusters vs subtypes · survival per cluster | Q5 structure in gene expression? Q6 do clusters match subtypes / prognosis? Q7 outliers? |

Shared by all pages: **header** (page tabs, patient counter, reset, help) and **filter sidebar** (subtype, receptors,
age, stage, grade, vital status, treatment, cohort-correction toggle, keep/exclude selection).

**Two interaction mechanisms** (the core of the grading item "linked views"):
- **Filter** (sidebar) = remove patients from every view.
- **Select** (lasso, box, click on a bar / cell / branch / cluster) = highlight patients in every view; the rest is greyed;
  charts can compare *selection vs rest*. A selection can be turned into a filter (Keep only / Exclude).

---

## 2. Decision log by phase

Format: **Decision**: why. *(Course link.)*

### Phase 0: foundations (`src/data.py`, `src/theme.py`)

**Repository layout.** `src/` (shared code), `notebooks/`, `data/` (dataset, not in the zip), `course_material/`,
`report/`, `docs/`. All paths are relative to the project root (brief requirement).

**One data layer (`load_data()`)** used by the notebook and the app, so both always see the same cleaned data.

| Decision | Why |
|---|---|
| Split the CSV into **clinical / mRNA / mutations** blocks | Three different kinds of data (mixed types, continuous z-scores, text codes) need different treatment. |
| **Mutations → one 0/1 flag per gene** (raw codes kept for tooltips) | The cells contain mutation codes (`H1047R`, `Q546H G1049R`). One-hot encoding them gives **7,757** near-empty columns. A biologist reasons at gene level ("is *TP53* mutated?"). |
| **No imputation** in the clinical block; missing categories become an explicit **"Unknown"** | The dashboard is for exploration: an imputed value would be shown as if real. The user sees how much is missing. |
| Keep **tumour stage** (26% missing) with "Unknown" | Clinically important; dropping the column (our first draft) or the rows (−26% of patients) loses too much. |
| `mutation_count` missing → **0** | Missing for exactly the 45 patients with no mutated gene in the panel (and r = 0.97 with the number of mutated genes elsewhere): missing means "none found", not "unknown". The only data-justified fill. |
| Fixes: `Positve` → `Positive`, 0/1 treatments → No/Yes, fixed category order | Consistency: every chart and filter lists categories the same way (grid 3.3). |
| Derived survival columns: `os_event`, `dss_event`, `vital_status` | Needed for Kaplan–Meier; see Phase 1. |
| `overall_survival_months` labelled **"Follow-up time (months)"** | For living patients it is time followed so far, not survival. |
| mRNA: **no extra scaling** | Every gene already has mean 0, sd 1 in this cohort (z-scores). Scaling again changes nothing. |

**Theme (`src/theme.py`).** One Plotly template for every chart + fixed colour meanings.

| Decision | Why | Course |
|---|---|---|
| PAM50 subtypes: **Okabe–Ito** colour-blind-safe palette with meaning (luminal = blues, Basal = vermillion) | Same colour = same subtype on every page; accessible | VAD Ch. 10 colour; grid 1.4, 4.5 |
| Validated palettes with a colour-vision-deficiency checker; ≤ 8 categorical colours, extra groups → "Other" grey | Human perception distinguishes only ~6–12 categorical colours | VAD Ch. 10 "limited number of discriminable bins" |
| Clusters use a **different** palette than subtypes | A cluster must never be mistaken for a subtype | |
| **Diverging** blue–grey–red for z-scores and correlations; **sequential** single blue for magnitudes; never rainbow | Diverging = sign matters (0 is meaningful); sequential = more/less | VAD Ch. 10 "rainbow is a poor default"; TP1 colour-scale reflection |
| z-score colours **clipped at ±3** | 0.93% of values are beyond ±3 (max 20.4); unclipped, they wash out the scale. Exact value stays in the tooltip | |
| Fix: Plotly's `simple_white` hides shapes (line width 0) → restored | Otherwise median lines, thresholds and the dendrogram cut were invisible | |

### Phase 1: exploratory analysis (`notebooks/Exploratory.ipynb`)

**Clinical relationships**
- **Spearman** (rank) correlation for numeric variables: tumour size and lymph nodes are very skewed; Spearman captures
  monotonic links and is robust to outliers. NPI ↔ nodes ρ = 0.73, NPI ↔ size 0.52 (expected: NPI is computed from them).
- **Normalized mutual information** for categorical variables (TP1). Normalised to [0, 1] so pairs with different numbers
  of categories are comparable. Strongest: PAM50 ↔ 3-gene classifier 0.32; PAM50 ↔ ER 0.29. Vital status ≤ 0.05 with
  any single variable: survival is not predictable from one feature.
- **Cross-table** PAM50 × integrative cluster: Basal → cluster 10 (69%), Her2 → cluster 5 (47%), luminal subtypes spread out.

**Survival**
- **Kaplan–Meier** curves, not "% deceased": follow-up time differs per patient (censoring).
- **Disease-specific** survival by default (death from breast cancer = event; death of another cause = censored), because
  480 patients died of other causes.
- Curves stop when **< 10 patients remain at risk** (otherwise one death makes the curve jump, down to 0% for the last patient).
- **Log-rank test** across subtypes: χ² = 94.4, df = 5, p ≈ 8 × 10⁻¹⁹. Basal and Luminal B curves **cross** after ~12 years.

**Genes and mutations**
- Sanity check: *GATA3* high in ER+, *ERBB2* high in HER2+ (the panel has no *ESR1* / *PGR* / *MKI67*).
- Basal-like: low luminal genes (*GATA3*, *BCL2*, *MAPT*, *ERBB3*), high cell-cycle genes (*CCNE1*, *AURKA*, *E2F3*).
  Luminal B vs A: higher proliferation.
- Mutations: *PIK3CA* 42% overall (57% of LumA); *TP53* 35% overall (88% of Basal, 70% of Her2); *GATA3* absent in Basal.

**Batch effect, the key finding**
- Colouring PCA by the **sampling cohort**: **PC1 is mostly cohort** (normalized MI 0.17 vs ≤ 0.04 for subtype/ER).
  The biology (ER, subtype, grade) is on PC2. This is a technical **batch effect**.
- Correction (`remove_cohort_effect`): centre each gene within its cohort, then re-standardise. Safe because the cohorts have
  the same subtype mix (normalized MI cohort ↔ PAM50 = 0.015). After correction: cohort ↔ PCs ≤ 0.01, PC1 = ER/subtype axis.
- **Dashboard: OFF by default** (data as published), with a sidebar toggle. The notebook uses the corrected data from §7.3 on.

**Dimensionality reduction** (on the 489 mRNA z-scores; clinical variables only colour the points)
- PCA: PC1 explains ~8%; 33 PCs for 50% of variance, 161 for 80%. No elbow: a 2-D PCA shows a small part of the structure.
- PCA, t-SNE (perplexity 30), UMAP (15 neighbours, min_dist 0.1) all show **one continuous cloud**, luminal → basal gradient,
  plus one small island (the 18-patient artefact).
- Subtype information is present but diffuse: a 15-NN classifier recovers PAM50 for 63–69% of patients (chance: 36%).
  Reason: the 489 genes are cancer-pathway genes, not the 50 PAM50 genes.
- **R_NX quality** (Lee & Verleysen; Lee's lecture): AUC t-SNE 0.272 > UMAP 0.189 > PCA 0.142 for small neighbourhoods,
  but PCA is best for large K (global structure). No single best projection.

**Clustering** (on the corrected 489-D data)
- **K-means, choosing k**: W decreases only ~13% from k = 2 to 10 (no elbow); B increases smoothly; Calinski–Harabasz is
  highest at k = 2 and decreases. → No well-separated clusters.
- k = 2 reproduces **ER+/luminal vs ER−/basal** (normalized MI with ER = 0.52).
- k = 6: three clusters dominated by one subtype; **purity 0.59** vs 0.36 baseline. Survival differs between clusters
  (χ² = 102.4, p ≈ 2 × 10⁻²⁰), as strongly as between subtypes.
- **Hierarchical**, 5 clusters: Ward [711, 528, 312, 212, 141]; complete [1727, 98, 60, 18, 1]; average and single
  collapse; **centroid** [1900, 1, 1, 1, 1] with **403 inversions**. → **Ward is the default**; the others stay available
  to show the effect.
- **DBSCAN**: in 489-D every patient is about equally far from its neighbours (k-distance curve flat for 97%); no eps gives
  more than one substantial cluster. On the UMAP embedding, results depend strongly on eps (17 clusters at 0.2, 7 at 0.25,
  1 + outliers from 0.4). In every setting it isolates the **18-patient group**.
- **The 18 patients**: all from cohort 1, mixed subtypes, extreme values (z ≈ +4 to +7) on a few unrelated genes
  (*SMAD5*, *MSH2*, *PPP2CB*…) → most likely a **technical artefact**. The dashboard lets the user select and exclude them.

**Removed from the notebook** (by us): the supervised-learning appendix (logistic regression, XGBoost, SHAP): not part of
the course and not needed by the dashboard.

### Phase 1b: UI design (sketch canvas v2)

| Decision | Why | Course / grid |
|---|---|---|
| **3 pages**, each ~one screen; header tabs | Avoid scrolling: the user must see the effect of a selection without scrolling | grid 4.4 |
| **Global filter sidebar** on every page, with counts next to options | Filters update every view; counts = "scented widgets" | brief §3.3; VAD Ch. 13 |
| **Linked selection** everywhere, bidirectional, selection vs rest | Core of exploration | brief §3.4; VAD Ch. 11/12 linked highlighting; Dash TP crossfiltering |
| Overview page first, details on demand (tooltips, i-buttons) | Shneiderman's mantra | VAD rules of thumb |
| Statistics drawn on charts (mean/median lines, KPIs), recomputed on filtered data | Brief requirement "compute and highlight properties" | brief §3.2 |
| Histogram **bin slider** | The bin size changes what you see | TP1 reflection; VAD Ch. 13 |
| **Pair view** adapts to variable types (cross-table / scatter / box) and accepts genes | One linked view replaces three static cards | |
| **Cluster heatmap** (heatmap + dendrogram) | Course idiom to assess clusters | VAD Ch. 7 |
| Heavy algorithms behind a **Run** button, spinner, cache; PCA precomputed | Responsiveness rule: 0.1 s / 1 s / 10 s | VAD rules of thumb; brief §4 |
| Cohort-correction toggle **off by default** | Show data as published; correction is an option the user understands | |
| **Trimmed** (outside the course, heavy to justify): Fisher/t-tests, volcano plot, oncoprint, silhouette, ARI, Cramér's V, CI band | Stay within the course; fewer, stronger views | course scope |

### Phase 2: app skeleton + Overview page (`app.py`, `pages/`, `src/state.py`, `src/figures.py`, …)

**Architecture**

| Decision | Why |
|---|---|
| **Plain Dash + CSS** (no extra UI library) | Same toolbox as the course's Dash session; fewer dependencies for the graders. |
| **Dash Pages** (`pages/overview.py`, `relationships.py`, `clusters.py`) | One module per page; the URL changes per page (back button works). |
| Header, sidebar and selection bar in the **app shell**, outside the pages | Filters and selection persist when changing page. |
| Shared state = 3 **`dcc.Store`s of patient ids**: `filtered-ids`, `selected`, `manual-filter` | Only lists of ids travel between browser and server, never data frames: every callback stays fast. |
| Pure **figure builders** (`src/figures.py`) separate from callbacks | Testable without a browser, reusable across pages. |
| Several callbacks write the selection (`allow_duplicate=True`) | Any chart can create a selection: one shared store, many writers. |

**Overview page and interactions**

| Feature | Interaction | Course / grid |
|---|---|---|
| KPI tiles (patients, median age, 5-year disease-specific survival, % died of breast cancer, median follow-up), with the selection's value underneath | recomputed on every filter / selection | brief §3.2 "compute properties"; grid 2.1 |
| Patients per subtype (bars) | **click a bar → select** that subtype; with a selection, pale = all, solid = selected | linked highlighting |
| Kaplan–Meier | split by any categorical variable (≤ 8 groups) or **selection vs rest**; disease-specific / overall; log-rank as text; groups < 10 patients left out (said under the chart) | |
| Distribution | histogram or box; mean/median lines; log scale (log(1 + x), ticks labelled in real values); **bin slider**; **drag a range → select**; with a selection both groups shown as **% of their own group** to compare shapes | TP1 bins; VAD "eyes beat memory" |
| Categorical breakdown | 100% stacked bars, all vs selection; **click a segment → select** | |
| Sidebar | subtype checklist with **live counts** (scented widget), receptors, age range, stage, grade, vital status, treatments, cohort-correction toggle; stays visible while scrolling | VAD scented widgets; grid 4.1 |
| Selection bar (under the header, only when something is selected) | "Keep only these" / "Exclude these" turn the selection into a filter; "Clear selection" | grid 4.3 feedback |
| (i) buttons, Help panel, hover templates | details on demand | grid 2.2, 4.2 |
| Loading indicator after 300 ms | feedback without flicker | grid 4.3 |

**Change vs the sketch:** *Keep only / Exclude* moved from the sidebar to the selection bar: they only make sense when
something is selected, so they appear exactly then.

**Measured speed** (full cohort, with a selection): filters 2 ms; each Overview chart ≤ 0.1 s → within the
"immediate response" class (VAD rules of thumb).

**Teacher questions for Phase 2**

*How do your views communicate?* Through three stores of patient ids. Each chart listens to `filtered-ids` and
`selected`; clicking or brushing a chart writes a new `selected`; the sidebar writes `filtered-ids`. Every view
recomputes from those ids, so linking is bidirectional by construction.

*Why ids and not the filtered data in the store?* A store is serialised to JSON and sent to the browser on every
update: 1,904 ints is tiny, a data frame with 700 columns is not. The data stays on the server, loaded once.

*Why normalise the histogram when a selection exists?* 87 selected patients next to 1,904 would be invisible on a count
scale; as "% of group" their shapes are directly comparable (side by side on a common scale, not from memory).

### Phase 3: Relationships & genomics page (`pages/relationships.py`, `src/figures_relationships.py`, `src/variables.py`)

**Four linked cards (2 × 2, fits one screen)**

| Card | What it shows | Interaction | Course |
|---|---|---|---|
| **Association matrix** | lower triangle; *categorical*: normalized mutual information between 9 clinical variables; *numeric*: Spearman ρ between 6 measurements | **click a cell → that pair opens in the pair view**; the pair shown is outlined | TP1 (correlation + MI heatmaps) |
| **Pair view** | any two variables: clinical, any of the 489 **genes** (expression) or any gene's **mutation status**; the chart follows the types: **cross-table** (cat × cat, row %), **scatter** (num × num, coloured by a categorical variable), **box plots** (num × cat) | click a cell / a box, or drag a box on the scatter → **selection**; with a selection: scatter = selected in colour, rest grey; boxes = selected patients as points over the group boxes | TP1 (scatter, box); VAD linked highlighting |
| **Mutation frequency** | top 8/12/16 most mutated genes among the filtered patients; *selection vs rest* (grouped bars, common scale) or *by subtype* (heatmap with an "All" column) | click → select the mutated patients (in that subtype) | |
| **Cluster heatmap** | mean expression per group (PAM50 subtypes, or selection vs rest) for the 10/15/25 genes that differ most; rows reordered by a **Ward dendrogram** of the genes | **click a gene → pair view shows that gene × PAM50** | VAD Ch. 7 cluster heatmap |

**Design decisions**

| Decision | Why |
|---|---|
| One **variable registry** (`clin:` / `gene:` / `mut:` keys, `src/variables.py`) | Every chart treats a variable the same way (type, label with units, category order); the Clusters page will reuse it for "colour by". |
| The pair view **chooses the chart from the types** | The user thinks in variables, not chart types; it replaces three static cards of the first sketch. Default: PAM50 × integrative cluster. |
| **Genes ranked by between-group variation B / total** (per gene) | Lecture 06's *B*, applied to one gene: the share of its variation explained by the groups. Replaces the ANOVA F-test of the notebook (outside the course). |
| "Unknown" values **left out** of mutual information | "Unknown" is not information. ⚠️ Because of this, the dashboard's values are slightly higher than the notebook's (PAM50 ↔ 3-gene 0.38 vs 0.32, the 3-gene classifier has 204 Unknown). Same ranking. |
| Cohort-correction toggle applies to **every gene-expression view** (pair view, heatmap) | One switch, consistent everywhere; the note under the chart says when it is on. |
| Cell values printed in the heatmap only up to 15 genes | Above that, rows are too thin to read the numbers (data-ink, readability). |
| Ward to order the **genes** in the heatmap | Same reason as for patients (balanced, no inversions); here it only orders rows, the groups stay fixed. |

**Measured speed** (full cohort, with a selection): 0.1–0.4 s per card (the 36 mutual informations of the matrix
take the longest).

**Teacher questions for Phase 3**

*Why does the pair view change chart type?* Because the right encoding depends on the data types (VAD: expressiveness):
two categories → counts in a table, two quantities → positions on two axes, one of each → distribution per group.

*Why row percentages in the cross-table?* To compare how each subtype is spread over the other variable regardless of
subtype size (LumA has 679 patients, Normal 140). Counts are in the tooltip.

*How do you choose the genes of the heatmap?* For each gene, the between-group variation B (as in the CH index of
lecture 06) divided by its total variation; we keep the genes with the largest share. The dendrogram then groups
genes with similar profiles: here a proliferation block (*E2F2, AURKA, CDK1, CCNE1*…) high in Basal, and a luminal block
(*GATA3, MAPT, BCL2, ERBB3*…) high in LumA/LumB.

*Is the heatmap a cluster heatmap?* Yes in Munzner's sense (Ch. 7): a heatmap whose rows are reordered by a hierarchical
clustering, with the dendrogram drawn next to it; its task is to judge whether groups differ.

*Why no statistical test on mutations?* The comparison is visual, on a common scale, which is the course's focus; with
many genes, p-values would need multiple-testing correction, which is outside the scope.

---

## 3. Course mapping

| Technique | Where used | Course source | In course? | If outside: justification |
|---|---|---|---|---|
| Bar, histogram, box, scatter, scatter matrix, heatmap | everywhere | TP1; VAD Ch. 7, 13 | ✅ | |
| Pearson correlation heatmap | (Spearman variant) | TP1 | ✅ | Spearman = Pearson on ranks; chosen for skewed variables |
| Mutual information | association matrix | TP1 | ✅ | normalized version for comparability |
| K-means (Lloyd), multiple inits | Clusters page | TP2; lecture 04 | ✅ | |
| Choosing k: W, B, CH index | Clusters page | TP3; lecture 06 | ✅ | |
| Hierarchical: single, complete, average, centroid; dendrogram, cut | Clusters page | TP3; lectures 05–06 | ✅ | |
| **Ward linkage** (default) | Clusters page | not in slides | ⚠️ | merges the pair that increases W least = greedy version of the K-means objective (lecture 04); the course linkages degenerate on this data |
| DBSCAN, eps heuristic (k-distance) | Clusters page | TP3 | ✅ | |
| Purity | clusters vs subtypes | TP3 asks "compute accuracy" | ✅ (spirit) | simple, explainable |
| PCA, scree, cumulative variance | Clusters page | TP5; DR lecture | ✅ | |
| t-SNE, UMAP | Clusters page | TP6; DR lecture | ✅ | |
| R_NX / Q_NX quality, AUC | Clusters page | TP6; DR lecture (Lee's own method) | ✅ | |
| Curse of dimensionality / norm concentration | explains DBSCAN, linkages | DR lecture | ✅ | |
| Linked views, filtering, aggregation, colour rules, rules of thumb | whole app | VAD (Munzner) | ✅ | |
| Cluster heatmap (heatmap + dendrogram) | Relationships page | VAD Ch. 7 | ✅ | |
| Between-group variation B (gene ranking) | Relationships heatmap | lecture 06 (CH index) | ✅ | applied per gene |
| **Kaplan–Meier + log-rank** | Overview, Clusters | not in course | ⚠️ | survival is the dataset's main outcome; a KM curve is derived data shown as superimposed lines (VAD); "% deceased" would be misleading with censoring |
| **Spearman** | association matrix | not explicit | ⚠️ | small step from TP1's Pearson; justified by skewness |
| **Cohort (batch) correction** | sidebar toggle | not in course | ⚠️ | found *with* a course tool (PCA coloured by a label); DR lecture: "appropriate data preprocessing" |

---

## 4. Teacher questions and answers

### Data and preprocessing

**Is the dataset complete? What did you do with missing values?**
Only the clinical block has missing values (tumour stage 26%, 3-gene classifier 11%, others < 6%). We do not impute:
categories get an explicit "Unknown", numbers stay missing and are left out chart by chart. The only fill is
`mutation_count`, missing exactly for the 45 patients without any mutated gene, so it is 0. Clustering and DR use the
mRNA block, which is complete.

**How did you handle categorical values?**
Kept as labelled categories with a fixed order (ordinal ones like grade and stage in their natural order). Mutation codes
became one binary flag per gene instead of 7,757 one-hot columns. We never feed one-hot clinical variables to the
clustering: it runs on gene expression only, clinical variables are used to colour and compare.

**Why didn't you standardise the genes?**
They are already z-scores: every gene has mean 0 and sd 1 in this cohort.

**What is the batch effect and how do you know it is technical?**
PC1 of the raw expression separates the sampling cohorts (NMI 0.17) while carrying almost no subtype/ER information.
Cohorts have the same subtype mix (NMI 0.015), so a cohort difference in expression cannot be biology. Also the 18-patient
artefact is entirely from cohort 1.

**Doesn't the correction remove real biology?**
It would if a cohort were enriched in one subtype; here the mix is the same. After correction the subtype information is
fully kept (15-NN accuracy 68% → 69%) and PC1 becomes the ER/subtype axis. The proper published method, ComBat, also
adjusts variances with shrinkage; our mean-centring is its simplest form, enough with large balanced batches (167–734
patients). It is off by default in the app so the user sees both.

### Visualization design

**Why these chart types?** Bars for counts per category, histogram/box for distributions (box when comparing groups),
scatter for two numeric variables, heatmaps for matrices (associations, cross-tables, expression), lines for survival.
Position on a common scale is the most accurate channel (VAD Ch. 5), so most comparisons use aligned bars or positions.

**Why a lower-triangle matrix?** It is symmetric; the upper triangle is redundant and the diagonal is always 1 (it would
dominate the colour scale). Better data-ink ratio (grid 1.5).

**Why diverging for correlations but sequential for mutual information?** Correlation has a sign and a meaningful 0;
mutual information is non-negative (more/less only).

**Why clip the z-score colour scale?** A few values reach 20; without clipping, 99% of cells would look grey.

**How is the design accessible?** Colour-blind-safe validated palettes; colour is never the only cue (legends, labels,
hover); text contrast ≥ 4.5:1; real buttons and labels; font ≥ 12 px.

**Why 3 pages and not one long page?** Grid 4.4: no excessive scrolling, and selections must show their effect without
scrolling. Each page answers specific user questions.

### Interaction and performance

**What is the difference between a filter and a selection?** A filter removes patients from all views; a selection
highlights them in all views (rest in grey) and enables "selection vs rest" comparisons. A selection can become a filter.

**How do you keep the interface responsive?** Data loaded once and cached; stores hold patient IDs, not data frames;
WebGL scatter plots; PCA precomputed; heavy algorithms (t-SNE, UMAP, hierarchical) only on "Run", with a spinner, and
cached per parameter set and filter; VAD's 0.1 s / 1 s / 10 s responsiveness classes.

**How would you scale to 100× more patients?** Approximate nearest neighbours / FIt-SNE / openTSNE, parametric UMAP,
mini-batch K-means, aggregated (binned) scatter plots instead of points, server-side caching (e.g. Redis), precomputing
embeddings offline (DR lecture: O(N²) methods need tree-based or multipole approximations).

### Survival

**Why Kaplan–Meier instead of a bar chart of deaths?** Censoring: a patient alive after 2 years of follow-up and one alive
after 20 years would count the same. KM uses the time each patient was observed.

**Why disease-specific survival?** 480 patients died of other causes; that says little about their tumour.

**What does the log-rank test assume?** Non-informative censoring; it is most powerful when hazards are proportional.
With crossing curves (Basal vs Luminal B) it loses power, which is why we show the curves, not only p.

**Why do the curves stop before the end?** Below 10 patients at risk, each death makes a large jump (to 0% at the last
patient); the tail would be misleading.

### Dimensionality reduction

**Why PCA first?** Linear, fast, deterministic, interpretable axes and loadings; Lee's lecture: "always try PCA first". It
revealed the batch effect.

**PCA vs t-SNE vs UMAP?** PCA keeps global variance; t-SNE keeps local neighbourhoods (perplexity ≈ neighbourhood size)
and can magnify or create cluster gaps; UMAP is similar to t-SNE, faster, with a controversial claim about global structure
(Lee). Axes of t-SNE/UMAP have no meaning; distances between far clusters are not reliable.

**How do you know a projection is faithful?** R_NX(K): for each neighbourhood size K, the share of each point's K nearest
neighbours preserved in 2-D, rescaled so random = 0 and perfect = 1; the AUC with log-weighting in K summarises it. Ours:
t-SNE best locally, PCA best globally, all values modest → the dashboard shows the curve next to the projection.

**Do you see clusters in the projections?** One continuous cloud with a luminal → basal gradient, plus the 18-patient
island. Consistent with the clustering metrics.

**Which parameters can the user change?** t-SNE perplexity; UMAP number of neighbours and min_dist. These control the
local/global trade-off.

### Clustering

**Explain K-means.** Choose k centres; assign each point to the nearest centre; move each centre to the mean of its
points; repeat until stable. It minimises the within-cluster variation W but only locally, so we run it 10 times from
random starts and keep the best (lecture 04).

**How did you choose k?** W, B and the Calinski–Harabasz index (ratio of B to W corrected for k). No elbow; CH is highest
at k = 2 and decreases → no well-separated groups; k = 2 is the ER split. Lecture 06 notes that choosing k is an open
problem, with the example "2 vs 3 breast-cancer subtypes".

**What is purity and its limitation?** Share of patients whose cluster's most frequent subtype is their own. It grows with
k (one cluster per patient gives 1), so compare at equal k and with the one-cluster baseline (0.36).

**Why Ward and not centroid, which the lecture calls the standard in biology?** On this data centroid linkage puts 1,900 of
1,904 patients in one cluster and its tree has 403 inversions (lecture 06's warning: such a tree cannot be cut at a
height). Average and single linkage chain the same way (lecture 05). Ward merges the two clusters whose union increases W
least, which is the K-means criterion applied greedily, and gives balanced clusters without inversions.

**Why do single/average/centroid fail here?** In 489 dimensions all pairwise distances are similar (norm concentration,
DR lecture), so linkages based on single distances chain everything.

**How does DBSCAN work and how did you choose eps?** A point with ≥ min_samples neighbours within eps is a core point;
clusters are connected core points; the rest is noise. eps from the knee of the sorted k-distance plot (with
min_samples = m, scikit-learn counts the point itself, so use k = m − 1). In 489-D there is no density structure; on UMAP it
mostly detects outliers and is very sensitive to eps (shown in the dashboard).

**What can clusters be used for?** Here: prognosis (clusters separate survival), checking subtypes (cluster × subtype
table), and finding artefacts (the 18 patients).

**Can you cross-filter with hierarchical clustering?** Yes: clicking a dendrogram branch selects its patients in all views;
dragging the cut line changes the number of clusters.

---

## 5. Key numbers cheat sheet

| Fact | Value |
|---|---|
| Patients / clinical cols / genes / mutation genes | 1,904 / 31 / 489 / 173 |
| One-hot mutation columns avoided | 7,757 |
| Missing: tumour stage / 3-gene / laterality | 26.3% / 10.7% / 5.6% |
| PAM50: LumA, LumB, Her2, Basal, claudin-low, Normal, NC | 679, 461, 220, 199, 199, 140, 6 |
| Died of breast cancer / of other causes / living | 622 / 480 / 801 (+1 unknown) |
| 5-year disease-specific survival: LumA / Basal | 93% / 64% |
| Log-rank across subtypes | χ² 94.4, df 5, p ≈ 8 × 10⁻¹⁹ |
| Spearman NPI ↔ nodes / size | 0.73 / 0.52 |
| NMI PAM50 ↔ 3-gene / ↔ ER; cohort ↔ PAM50 | 0.32 / 0.29; 0.015 |
| PCA: PC1 variance; PCs for 50% / 80% | ~8%; 33 / 161 |
| NMI PC1 ↔ cohort before / after correction | 0.17 / 0.00 |
| R_NX AUC t-SNE / UMAP / PCA | 0.272 / 0.189 / 0.142 |
| K-means k = 2 NMI with ER; k = 6 purity (baseline) | 0.52; 0.59 (0.36) |
| Survival by K-means k = 6 | χ² 102.4, p ≈ 2 × 10⁻²⁰ |
| Centroid linkage inversions | 403 |
| Outlier group | 18 patients, all cohort 1 |

---

## 6. Known limitations

- The gene panel lacks the PAM50 genes and classic markers (*ESR1*, *PGR*, *MKI67*), so subtype separation is weaker than
  in the literature.
- Batch correction by mean-centring is simpler than ComBat.
- Kaplan–Meier/log-rank compare groups; they do not adjust for confounders (age, stage) as a Cox model would.
- t-SNE/UMAP are stochastic (fixed seed for reproducibility) and their large-scale layout is not reliable.
- Purity depends on k; mutual information values are low in absolute terms (0.3 is already strong).
- DBSCAN on a UMAP embedding inherits UMAP's distortions.

---

## 7. Glossary

| Term | Meaning |
|---|---|
| **PAM50** | Gene-expression classifier of breast tumours into intrinsic subtypes (Luminal A/B, HER2-enriched, Basal-like, Normal-like; plus claudin-low here). |
| **ER / PR / HER2** | Hormone receptors (oestrogen, progesterone) and the HER2 growth-factor receptor; drive treatment choices. |
| **Integrative cluster** | Another molecular classification (11 groups) from METABRIC, combining expression and copy number. |
| **NPI** | Nottingham Prognostic Index = 0.2 × tumour size (cm) + node stage + grade; higher = worse. |
| **z-score** | (value − mean) / sd of the gene across patients; 0 = average expression. |
| **Censoring** | The event (death) has not been observed by the end of follow-up; we only know survival ≥ follow-up time. |
| **Batch effect** | Technical difference between groups of samples processed separately. |
| **Linkage** | Rule giving the distance between two clusters in hierarchical clustering. |
| **Inversion** | A dendrogram merge drawn lower than the merge before it (centroid linkage). |
| **R_NX(K)** | Rescaled share of K-nearest neighbours preserved by a projection (0 = random, 1 = perfect). |
| **Purity** | Share of points whose cluster's majority label is their own label. |
| **Normalized mutual information** | Mutual information divided by the variables' entropies, in [0, 1]. |
