# METABRIC Explorer: study guide

*For Renan and Lays. Last updated: code reorganisation (2026-11-23). Update it at the end of every phase.*

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
8. [Must go in the report](#8-must-go-in-the-report)

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
| **1. Overview** | KPI tiles · patients per subtype · survival over the first 5 years · distribution of a variable · categorical breakdown | Q1 what does my (filtered) cohort look like? Q3 does survival differ between groups? |
| **2. Relationships & genomics** | association matrix (mutual information / Spearman) · adaptive pair view · mutation frequency · cluster heatmap | Q2 how are variables linked? Q4 which genes / mutations differ in a group? |
| **3. Clusters & projections** | settings + Run · projection scatter · quality (choose-k, R_NX, dendrogram) · clusters vs subtypes · survival per cluster | Q5 structure in gene expression? Q6 do clusters match subtypes / prognosis? Q7 outliers? |

Shared by all pages: **header** (page tabs, patient counter, reset, help) and **filter sidebar** (subtype, receptors,
age, stage, grade, vital status, treatment) and **selection bar** (keep only / exclude / clear).

**Two interaction mechanisms** (the core of the grading item "linked views"):
- **Filter** (sidebar) = remove patients from every view.
- **Select** (lasso, box, click on a bar / cell / branch / cluster) = highlight patients in every view; the rest is greyed;
  charts can compare *selection vs rest*. A selection can be turned into a filter (Keep only / Exclude).

---

## 2. Decision log by phase

Format: **Decision**: why. *(Course link.)*

### Phase 0: foundations (`src/data.py`, theme section of `src/figures.py`)

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
| Derived survival columns: `os_event`, `dss_event`, `vital_status` | Needed for the survival chart; see Phase 1. |
| `overall_survival_months` labelled **"Follow-up time (months)"** | For living patients it is time followed so far, not survival. |
| mRNA: **no extra scaling** | Every gene already has mean 0, sd 1 in this cohort (z-scores). Scaling again changes nothing. |

**Theme (first section of `src/figures.py`).** One Plotly template for every chart + fixed colour meanings.

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
- **Share alive per month**, not "% deceased": follow-up time differs per patient (censoring). For each month t, the
  share of patients still alive among those whose status at t is known (see §4 "Our own questions").
- Shown over the **first 5 years** only: after that, patients recruited late are missing from the share and the curve
  falls too fast.
- **Disease-specific** survival by default (death from breast cancer = event; death of another cause is not an event),
  because 480 patients died of other causes.
- A curve stops when **< 10 patients have a known status**.
- Alive at 5 years: LumA 93%, LumB 80%, claudin-low 82%, Normal 84%, Her2 67%, Basal 63%.

**Genes and mutations**
- Sanity check: *GATA3* high in ER+, *ERBB2* high in HER2+ (the panel has no *ESR1* / *PGR* / *MKI67*).
- Basal-like: low luminal genes (*GATA3*, *BCL2*, *MAPT*, *ERBB3*), high cell-cycle genes (*CCNE1*, *AURKA*, *E2F3*).
  Luminal B vs A: higher proliferation.
- Mutations: *PIK3CA* 42% overall (57% of LumA); *TP53* 35% overall (88% of Basal, 70% of Her2); *GATA3* absent in Basal.

**Dimensionality reduction** (on the 489 mRNA z-scores; clinical variables only colour the points)
- PCA: PC1 explains ~8%; 33 PCs for 50% of variance, 161 for 80%. No elbow: a 2-D PCA shows a small part of the structure.
- PCA, t-SNE (perplexity 30), UMAP (15 neighbours, min_dist 0.1) all show **one continuous cloud**, luminal → basal gradient,
  plus one small island (the 18-patient artefact).
- Subtype information is present but diffuse: a 15-NN classifier recovers PAM50 for 63–69% of patients (chance: 36%).
  Reason: the 489 genes are cancer-pathway genes, not the 50 PAM50 genes.
- **R_NX quality** (Lee & Verleysen; Lee's lecture): AUC t-SNE 0.272 > UMAP 0.189 > PCA 0.142 for small neighbourhoods,
  but PCA is best for large K (global structure). No single best projection.

**Clustering** (notebook, on the corrected 489-D data; the dashboard uses the published data, so its numbers differ)
- **K-means, choosing k**: W decreases only ~13% from k = 2 to 10 (no elbow); B increases smoothly; Calinski–Harabasz is
  highest at k = 2 and decreases. → No well-separated clusters.
- k = 2 reproduces **ER+/luminal vs ER−/basal** (normalized MI with ER = 0.52).
- k = 6: three clusters dominated by one subtype; **purity 0.59** vs 0.36 baseline. 5-year survival per cluster
  ranges from 65% to 92%, as wide as between subtypes (63% to 93%).
- **Hierarchical**, 5 clusters: Ward [711, 528, 312, 212, 141]; complete [1727, 98, 60, 18, 1]; average and single
  collapse; **centroid** [1900, 1, 1, 1, 1] with **403 inversions**. → **Ward is the default**; the others stay available
  to show the effect.
- **DBSCAN**: in 489-D every patient is about equally far from its neighbours (k-distance curve flat for 97%); no eps gives
  more than one substantial cluster. On the UMAP embedding, results depend strongly on eps (17 clusters at 0.2, 7 at 0.25,
  1 + outliers from 0.4). In every setting it isolates the **18-patient group**.
- **The 18 patients**: all from cohort 1, mixed subtypes, extreme values (z ≈ +4 to +7) on a few unrelated genes
  (*SMAD5*, *MSH2*, *PPP2CB*…) → most likely a **technical artefact**. The dashboard lets the user select and exclude them.


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

### Phase 2: app skeleton + Overview page (`app.py`, `pages/`, `src/data.py`, `src/figures.py`, `src/layout.py`)

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
| Survival over the first 5 years | share alive per month (months 0 to 60); split by any categorical variable (≤ 8 groups) or **selection vs rest**; disease-specific / overall; 5-year values as text under the chart; groups < 10 patients left out (said under the chart) | TP1 line chart; Kaplan–Meier until 2026-11-23 |
| Distribution | histogram or box; mean/median lines; log scale (log(1 + x), ticks labelled in real values); **bin slider**; **drag a range → select**; with a selection both groups shown as **% of their own group** to compare shapes | TP1 bins; VAD "eyes beat memory" |
| Categorical breakdown | 100% stacked bars, all vs selection; **click a segment → select** | |
| Sidebar | subtype checklist with **live counts** (scented widget), receptors, age range, stage, grade, vital status, treatments; stays visible while scrolling | VAD scented widgets; grid 4.1 |
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

### Phase 3: Relationships & genomics page (`pages/relationships.py`, `src/figures.py`, `src/data.py`)

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
| One **variable registry** (`clin:` / `gene:` / `mut:` keys, in `src/data.py`) | Every chart treats a variable the same way (type, label with units, category order); the Clusters page will reuse it for "colour by". |
| The pair view **chooses the chart from the types** | The user thinks in variables, not chart types; it replaces three static cards of the first sketch. Default: PAM50 × integrative cluster. |
| **Genes ranked by between-group variation B / total** (per gene) | Lecture 06's *B*, applied to one gene: the share of its variation explained by the groups. Replaces the ANOVA F-test of the notebook (outside the course). |
| "Unknown" values **left out** of mutual information | "Unknown" is not information. ⚠️ Because of this, the dashboard's values are slightly higher than the notebook's (PAM50 ↔ 3-gene 0.38 vs 0.32, the 3-gene classifier has 204 Unknown). Same ranking. |
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

### Phase 4: Clusters & projections page (`pages/clusters.py`, `src/analysis.py`, `src/figures.py`)

**Cards**

| Card | What it does | Interaction | Course |
|---|---|---|---|
| **Settings + Run** | projection PCA / t-SNE (perplexity) / UMAP (neighbours, min_dist); clustering K-means (k) / hierarchical (linkage, k) / DBSCAN (eps percentile, min_samples); cluster in the 489 genes or in the 2-D view | nothing heavy runs until **Run**; status line gives method, clusters, noise, time, warnings | TP2, TP3, TP5, TP6; brief §4 |
| **Projection** | 2-D scatter; colour by cluster, any clinical variable, any gene (diverging scale) or mutation; PCA axes show % variance, t-SNE/UMAP axes hidden | **lasso / box → selection** on every page | DR lecture; VAD linked highlighting |
| **Diagnostics** | *Choose k*: W, B, Calinski–Harabasz for k = 2..8 (current k circled); for DBSCAN the sorted **k-distance curve** with eps; *Quality*: **R_NX(K)** + AUC; *Dendrogram*: top 30 branches, cut line, **inversion warning** | **click a branch → select its patients** | TP3, lectures 05–06, TP6 |
| **Clusters vs subtypes** | row % cluster × PAM50 + a "whole cluster" column with sizes; **purity** vs one-cluster baseline | click → select the cluster or cluster ∩ subtype | TP3 "accuracy" |
| **Survival per cluster** | share alive per month over the first 5 years, one line per cluster | | "what are clusters useful for?" |

**Clusters are shared app-wide**: after a Run, the Overview survival chart can split **by cluster**, and the mutation and
cluster-heatmap cards of page 2 have a **"By cluster"** mode (what does each cluster mean biologically?).

**Design decisions**

| Decision | Why |
|---|---|
| **Run button**, nothing heavy on slider moves; spinner during the run | VAD responsiveness classes: > 1 s needs explicit feedback; the brief suggests forcing a choice before heavy algorithms. |
| **Cache** (`functools.lru_cache`) per method + parameters + patients | Going back to a previous setting is instant (re-run t-SNE: 12.5 s → 0.0 s). |
| Runs use the **filtered patients at the time of the Run**; if filters change, the projection keeps the computed patients that still match and says "press Run" | No surprise 30-s recomputation when a filter is clicked; the user stays in control. |
| **UMAP warm-up** in a background thread at start-up | UMAP compiles its code on first use (numba); this moves ~15–20 s out of the first Run. |
| **DBSCAN eps as a percentile of the k-distance curve** (default 90th), k = min_samples − 1 | TP3's heuristic made interactive; works in any space (489-D distances ≈ 20–50, UMAP ≈ 0.1–0.5) without retuning the slider. |
| DBSCAN defaults to the **2-D view**; K-means / hierarchical to the **489 genes** | In 489-D no density structure exists (curse of dimensionality); K-means/Ward use all the information. |
| Clusters named **C1 = largest**, noise grey; cluster palette ≠ subtype palette | Stable, readable names; a cluster is never mistaken for a subtype. |
| Dendrogram **truncated to 30 branches**; branch = merge of Z matched by height | 1,904 leaves are unreadable; every visible dot is a clickable group of patients. |
| Cut chosen with the **k slider** (not by dragging the line) | Simpler and exact; the dashed line shows where the cut falls. |

**Measured speed** (1,904 patients): PCA + K-means 0.5 s; t-SNE + Ward 12.5 s; UMAP + DBSCAN 32 s on the very first run
(numba compilation, now pre-warmed), ~12 s after; centroid tree 0.8 s; choose-k 2 s; R_NX 2.4 s; any cached setting
instant; charts redrawn after a run ≤ 0.15 s.

**Teacher questions for Phase 4**

*Which parameters can the user change, and why those?* t-SNE perplexity and UMAP neighbours/min_dist (local vs global
trade-off); k for K-means and hierarchical; the linkage; eps and min_samples for DBSCAN; the space clustered. These are
exactly the parameters whose effect the course asks to explore ("how does changing the parameters change the
clustering?"). Everything else is fixed (seed 42) for reproducibility.

*Can a clustering help choose the DR method?* Yes: colour the projection by the clusters found in 489-D. If a projection
keeps clusters together it preserves the structure the clustering found; R_NX measures the same thing quantitatively.

*Do you see visual clusters in the projections?* Not really: one continuous cloud with the luminal → basal gradient;
t-SNE magnifies small gaps (lecture: t-SNE can show spurious clusters), so we judge with R_NX and the cluster metrics,
not by eye.

*How do you cross-filter with hierarchical clustering?* Click a branch of the dendrogram: its patients become the
selection on every page. The composition table and the survival chart then describe that branch.

*What happens with centroid linkage?* The status line and the dendrogram warn: ~400 inversions; the tree is a staircase
(chaining), cutting it gives one giant cluster. This is lecture 06's drawback, visible live.

*How does the interface stay responsive?* Heavy work only on Run, cached results, pre-warmed UMAP, WebGL scatter,
ids-only stores, and the other charts never wait for a heavy computation.

### Phase 5: polish against the evaluation grid, "simplicity first"

**Simplicity review against the course.** Everything the app does was re-checked against the practical sessions, the
clustering and DR lectures and Munzner's slides. Outside the course remain only the four justified items (Kaplan–Meier,
Spearman, Ward as default; the cohort correction and Kaplan–Meier were later removed). Three controls were **removed** because they added a concept without
adding insight:

| Removed | Replaced by |
|---|---|
| "Cluster in: 489 genes / 2-D view" (Clusters page) | a fixed rule, stated under the algorithm buttons and in the help: K-means and hierarchical on the 489 genes; DBSCAN on the 2-D view (as in TP3; curse of dimensionality) |
| "Group by" for the Overview box plots | box plots always compare the PAM50 subtypes, or selection vs rest when patients are selected |
| mutation options in the pair view's "Colour" list | colour by a categorical clinical variable only |

**Other fixes:** the Help panel was rewritten (filter → select → details, one paragraph per page, "good to know") and
now **opens by itself on the first visit only** (remembered in the browser); every chart text ≥ 11 px (heatmap capped
at 20 genes); "1 cluster" grammar; README note for Windows (`py` instead of `python`).

### Evaluation-grid checklist (use it for the report and the oral exam)

| Grid item | How we address it | Where |
|---|---|---|
| 1.1 Appropriate charts | bars for counts, histogram/box for distributions, scatter for 2 numerics, heatmaps for matrices/tables, lines for survival; the pair view picks the chart from the data types | all pages |
| 1.2 No chart junk | no 3-D, no shadows, light grid, no decorative colour; one accent colour | `figures.py` (theme) |
| 1.3 Readability | axis titles with units, legends, hover on every mark, notes under charts (n, statistics) | all charts |
| 1.4 Colours | colour-blind-validated palettes; fixed subtype colours; diverging only for signed values (z, ρ), sequential for magnitudes; z clipped at ±3 | `figures.py` (theme) |
| 1.5 Data-ink | lower-triangle matrices (no redundant half, no diagonal), values in cells only when readable | page 2 |
| 2.1 Linked views | one shared selection and one shared filter; every chart listens; 9 charts can create a selection; selection ↔ filter (Keep only / Exclude) | `data.py` (filters and selection), `layout.py` (callbacks) |
| 2.2 Details on demand | hover templates on every mark; (i) help per card; status lines | all pages |
| 2.3 Speed | ids-only stores; all light charts ≤ 0.4 s; heavy algorithms only on Run, cached, UMAP pre-warmed | `analysis.py` (cached computations) |
| 3.1 Axes & scales | y from 0 for counts, % and choose-k; survival 0–100%; log scale only on request, ticks in real units; PCA axes show % variance; t-SNE/UMAP axes hidden (meaningless) | figures |
| 3.2 Density vs simplicity | 3 pages × 4–5 cards; controls trimmed (Phase 5); heatmap ≤ 20 genes | layout |
| 3.3 Consistency | one Plotly template, one font, same colours/legends/selection style everywhere | `figures.py` (theme), `style.css` |
| 4.1 Navigation | 3 tabs, active tab highlighted; sidebar always visible; reset button | header, sidebar |
| 4.2 Guidance | guide on first visit + Help button; (i) per card; notes under charts say what to do | `layout.py` |
| 4.3 Feedback | loading indicator (after 300 ms), Run status line (time, warnings), selection bar, patient counter, "press Run" when outdated | all pages |
| 4.4 Layout | each page fits a 1440 × 1000 screen: a selection's effect is visible without scrolling | `style.css` |
| 4.5 Accessibility | colour-blind palettes, colour never the only cue (legends/labels), contrast ≥ 4.5:1, text ≥ 11 px in charts, real buttons/labels, keyboard focus outlines | `figures.py` (theme), `style.css` |
| 5.1 Creativity | adaptive pair view; cluster heatmap linked to the pair view; histograms normalised to compare a selection's shape; clusters reused on every page ("by cluster"); live dendrogram selection with inversion warning | pages 2–3 |
| 6.x Report | to do in Phase 6 (this guide is the source) | |

**Remaining gaps (say them if asked):** selecting inside a chart needs a mouse (Plotly charts are not keyboard-operable);
t-SNE/UMAP still take 10–30 s on 1,904 patients (mitigated by Run, spinner, cache, warm-up); the dashboard is designed
for a laptop/desktop screen (it stacks on narrow screens but is not optimised for phones).

### Phase 5, feedback round (after running the app locally)

| Feedback | Change |
|---|---|
| Clicking a subtype did not change the survival chart | a **new selection switches the survival chart to "selection vs rest"**; clearing it switches back to subtypes; the user can still pick another split |
| *Keep only / Exclude* did not update the sidebar | a selection made on a variable that has a sidebar control (subtype, ER/PR/HER2, stage, grade, vital status, treatments, age range) now **changes that control**: keep only Basal ticks only Basal, exclude ER+ sets ER to "−", keep age 40–50 moves the slider. Selections without a matching control (lasso, dendrogram branch, lymph-node range…) still become a list of kept/excluded patients, reported under the sidebar |
| Survival and distribution charts overflowed their cards | every chart has **one height used by both the container and the figure** (they had drifted apart) |
| "PAM50 subtype" was not accurate (claudin-low comes from another classifier) | renamed **"Molecular subtype (PAM50 + claudin-low)"** in axes, legends, dropdowns and help |
| The cohort (batch) correction was confusing | **removed from the dashboard** (sidebar toggle, corrected-data code paths, help text). The finding stays in the notebook, which now defines the correction itself |

**Consequence for numbers:** the notebook's clustering results (e.g. K-means k = 6 purity 0.59, 5-year survival 65–92%) were
computed on the *corrected* data; the dashboard works on the published data, so its values differ (e.g. K-means k = 4
purity 0.46, 5-year survival 65–86%). Quote the dashboard's values when demonstrating the dashboard.

### Survival without Kaplan–Meier (2026-11-23)

Kaplan–Meier and the log-rank test were not in the course and would have been hard to defend at the exam. They were
**replaced by a simple in-course measure**: for each month t, the share of patients still alive among the patients whose
status at t is known, drawn as a line chart over the **first 5 years**.

| Before | After |
|---|---|
| Kaplan–Meier curves up to 30 years, stopped at < 10 patients at risk | share alive per month, months 0 to 60, stopped at < 10 patients with a known status |
| log-rank χ², df, p under the chart | "Alive at 5 years: LumA 93% · …" under the chart |
| `stats.kaplan_meier`, `stats.median_survival`, `stats.logrank_test`, `figures.km_figure` | `analysis.survival_share`, `analysis.survival_at`, `figures.survival_figure` (old functions deleted) |
| KPI "5-year DSS" from Kaplan–Meier | KPI "Alive at 5 years" from the share |

The notebook (section 5 and the cluster-survival cell) was patched the same way. Up to 5 years the share agrees with
Kaplan–Meier within about one point (all patients: 81% vs 82%).

### Code reorganisation: 13 modules → 4 (2026-11-23)

To make the project easier to read, `src/` was merged into **four files, one per layer**:
data → computation → visualization → interface. Nothing in the dashboard's behaviour changed (same callbacks, same
numbers, checked before/after). Each file starts with a contents list and keeps one section per former module.

| New file | Former modules | What it holds |
|---|---|---|
| `src/data.py` | data, state, variables | loading and cleaning; filters and selection; the variable registry |
| `src/analysis.py` | stats, algorithms, compute | survival share, mutual information; DR, clustering, quality measures; cached computations |
| `src/figures.py` | theme, figures, figures_relationships, figures_clusters | colours and Plotly template; every chart builder, one section per page |
| `src/layout.py` | components, callbacks | header, sidebar, selection bar, cards, help; app-wide callbacks |

Renamed to avoid clashes: the cached versions in the Clusters page are `analysis.cached_linkage_tree` and
`analysis.cached_k_distance` (the plain `linkage_tree` / `k_distance` are the algorithms); the selection-bar callback is
`update_selection_bar`. An unused colour helper (`theme.cluster_colors`) was deleted.

*Why not 3 files?* Merging `layout.py` into `app.py` would force the pages to import `app.py`, which re-runs it and
creates a second Dash app. Four layers is the smallest clean split.

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
| Survival share per month (first 5 years) | Overview, Clusters | TP1 (proportions, line chart); VAD Ch. 2 (derived attribute) | ✅ | replaces Kaplan–Meier + log-rank (removed 2026-11-23) |
| **Spearman** | association matrix | not explicit | ⚠️ | small step from TP1's Pearson; justified by skewness |
| **Cohort (batch) correction** | notebook only | not in course | ⚠️ | found *with* a course tool (PCA coloured by a label); removed from the dashboard (too hard to interpret) |

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
patients). We kept it in the notebook only: in the dashboard it was a concept users could not easily act on.

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

**Why a curve over time instead of a bar chart of deaths?** Censoring: a patient alive after 2 years of follow-up and one
alive after 20 years would count the same. The curve uses, at each month, only the patients whose status is known.

**Why disease-specific survival?** 480 patients died of other causes; that says little about their tumour.

**Why only the first 5 years?** After that, the share is biased downwards (see "Our own questions" below). 5-year
survival is also the usual clinical summary.

**Why do some groups not appear?** A group needs at least 10 patients with a known status; smaller groups are listed
under the chart as "not shown".

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

### Our own questions (Renan, 2026-11-23), with answers

**What exactly is an mRNA z-score?** One value per patient and gene: the tumour's expression of that gene, standardised
over all 1,904 patients: z = (expression in this patient − mean of the gene) / standard deviation of the gene.
z = 0: like the average patient; z = +2: 2 SD above average (gene strongly "switched on"); z = −2: well below. Example:
*ERBB2* (the gene that makes HER2) averages +2.14 in HER2-positive patients and −0.30 in the others. A z-score is
relative to its own gene: "GATA3 z = 1" and "AURKA z = 1" both mean "above that gene's average", not "equally active".
This is also why no extra scaling is needed before PCA or clustering (every gene already has mean 0, SD 1).

**Why not Kaplan–Meier, and is our survival chart in the course?** Kaplan–Meier (used until 2026-11-23) is not in the
course, so we replaced it. The outcome of this dataset is *time to death*, and it is **censored** (801 patients were alive
at their last follow-up: we only know they survived *at least* that long); "% deceased" would count a patient followed 1
year like one followed 20 years. Our chart is a **proportion per month** (TP1-level statistics) drawn as **superimposed
line charts** (a course idiom), i.e. a **derived attribute** (Munzner Ch. 2).

**How is the survival share computed?** For each month t from 0 to 60:
*alive* = patients followed for more than t months; *dead* = patients who died (of breast cancer, by default) at or before
t; *known* = alive + dead; **share = alive / known**. A patient whose follow-up ended before t, still alive, is left out
at t (we do not know their status). Toy example, 5 patients: deaths at 10 and 30 months, censored at 20, 40, 50 →
at month 25: alive = 3 (30, 40, 50), dead = 1 (10), the one censored at 20 is left out → share = 3/4 = 75%.
Code: `analysis.survival_share`.

**Why stop at 5 years?** Patients were recruited over many years, so at long horizons the patients recruited later
(still alive, but not yet followed that long) drop out of the share while early deaths stay in, and the share falls too
fast. Comparison with Kaplan–Meier, all patients, disease-specific:

| Years | 2 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| Kaplan–Meier | 95% | 82% | 71% | 64% | 57% |
| Our share | 95% | 81% | 65% | 47% | 22% |

Up to 5 years almost every patient's status is known and the two agree within about one point; that is why the chart
shows months 0 to 60 only. Limitation: it compares groups visually; there is no test (the log-rank test was removed with
Kaplan–Meier) and no adjustment for age or stage.

**Why is Ward the default linkage?** Ward merges the two clusters whose union increases the within-cluster variation W
the least: the K-means objective of lecture 04, applied step by step. On our data it is the **only linkage that works**
(5 clusters, published data):

| Linkage | Cluster sizes | Problem |
|---|---|---|
| **Ward** | 710, 689, 387, 100, 18 | none: balanced, no inversions |
| Complete | ~90% in one cluster | crowding (lecture 05) |
| Average | one giant cluster + singletons | chaining |
| Single | 1,900 + four single patients | chaining (lecture 05) |
| Centroid | 1,900 + singletons, 442 inversions | cannot be cut at a height (lecture 06) |

The course linkages fail because in 489 dimensions all patients are about equally far apart (norm concentration, DR
lecture). Ward's own limits: like K-means it prefers compact, similar-size clusters and needs Euclidean distances (fine
for z-scores). Not in the slides, so it is justified, and the other linkages stay selectable to show why.

**Why PCA, t-SNE, UMAP and K-means, hierarchical, DBSCAN?** The brief imposes the categories (K-means + one hierarchical
+ one density-based; one linear + one non-linear projection); within them we used what the practical sessions teach.

| Method | Advantages | Disadvantages | On our data |
|---|---|---|---|
| PCA (linear, TP5) | fast, deterministic, meaningful axes (% variance, loadings); "always try PCA first" | only straight-line structure; PC1 = 8% | revealed the batch effect; best R_NX for large neighbourhoods |
| t-SNE (non-linear, TP6) | best at keeping close neighbours | slow (~12 s), meaningless axes, can exaggerate or invent gaps | best R_NX for small neighbourhoods (AUC 0.27) |
| UMAP (non-linear, TP6) | like t-SNE, scales better | slow first run (compilation), global layout "controversial" (Lee) | in between (AUC 0.19) |
| K-means (TP2) | simple, fast, required by the brief | k must be chosen; round clusters; random start | k = 2 recovers ER+/ER− |
| Hierarchical (TP3) | one tree gives every k; dendrogram → cross-filtering | O(n²) memory; linkage choice critical | only Ward works |
| DBSCAN (TP3) | no k; any shape; labels outliers | very sensitive to eps; useless in 489-D | isolates the 18-patient artefact |

We keep **both t-SNE and UMAP** to compare two non-linear methods (decision of 2026-11-23). **MDS (TP5) is not used**
because classical MDS with Euclidean distances gives exactly the same embedding as PCA (DR lecture), so it would be
redundant.

**Is claudin-low a PAM50 subtype?** Not strictly: claudin-low comes from a separate classifier (Prat et al., 2010).
METABRIC publishes the combination: the original column is `pam50_+_claudin-low_subtype` (five PAM50 subtypes plus
claudin-low as a sixth group; NC = not classified). We keep the data as published and label it honestly:
**"Molecular subtype (PAM50 + claudin-low)"** everywhere in the dashboard.

---

## 5. Key numbers cheat sheet

| Fact | Value |
|---|---|
| Patients / clinical cols / genes / mutation genes | 1,904 / 31 / 489 / 173 |
| One-hot mutation columns avoided | 7,757 |
| Missing: tumour stage / 3-gene / laterality | 26.3% / 10.7% / 5.6% |
| PAM50: LumA, LumB, Her2, Basal, claudin-low, Normal, NC | 679, 461, 220, 199, 199, 140, 6 |
| Died of breast cancer / of other causes / living | 622 / 480 / 801 (+1 unknown) |
| Alive at 5 years (disease-specific): all patients | 81% |
| Alive at 5 years: LumA / LumB / Her2 / Basal / claudin-low / Normal | 93 / 80 / 67 / 63 / 82 / 84% |
| Spearman NPI ↔ nodes / size | 0.73 / 0.52 |
| NMI PAM50 ↔ 3-gene / ↔ ER; cohort ↔ PAM50 | 0.32 / 0.29; 0.015 |
| PCA: PC1 variance; PCs for 50% / 80% | ~8%; 33 / 161 |
| NMI PC1 ↔ cohort before / after correction | 0.17 / 0.00 |
| R_NX AUC t-SNE / UMAP / PCA | 0.272 / 0.189 / 0.142 |
| K-means k = 2 NMI with ER; k = 6 purity (baseline) | 0.52; 0.59 (0.36) |
| Alive at 5 years per K-means k = 6 cluster (notebook) / k = 4 (dashboard) | 65–92% / 65–86% |
| Centroid linkage inversions | 403 |
| Outlier group | 18 patients, all cohort 1 |

---

## 6. Known limitations

- The gene panel lacks the PAM50 genes and classic markers (*ESR1*, *PGR*, *MKI67*), so subtype separation is weaker than
  in the literature.
- The dashboard shows expression as published: the cohort (batch) effect found in the notebook is not corrected there.
- The survival share is reliable only over the first 5 years (later, patients recruited late are missing and the curve
  falls too fast). It compares groups visually, with no test and no adjustment for confounders (age, stage).
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
| **Survival share** | At month t, patients alive / patients whose status at t is known (alive or already dead). |
| **Batch effect** | Technical difference between groups of samples processed separately. |
| **Linkage** | Rule giving the distance between two clusters in hierarchical clustering. |
| **Inversion** | A dendrogram merge drawn lower than the merge before it (centroid linkage). |
| **R_NX(K)** | Rescaled share of K-nearest neighbours preserved by a projection (0 = random, 1 = perfect). |
| **Purity** | Share of points whose cluster's majority label is their own label. |
| **Normalized mutual information** | Mutual information divided by the variables' entropies, in [0, 1]. |

---

## 8. Must go in the report

Points we decided must appear in the final report (in addition to the grid's 6.x items):

- **Data**: what a z-score is (one value per patient and gene, standardised over the cohort) and why no extra scaling.
- **Survival**: why a curve over time and not "% deceased" (censoring), the share alive / known with the toy example,
  why only the first 5 years (the Kaplan–Meier comparison table), framed as *derived data + superimposed line charts*.
- **Ward**: the linkage comparison table (sizes, inversions), the link to the K-means objective, and why the course
  linkages fail here (norm concentration).
- **Projections**: PCA vs t-SNE vs UMAP advantages / disadvantages, the R_NX comparison, and **why not MDS**
  (classical MDS with Euclidean distances = PCA, DR lecture).
- **Clustering**: K-means vs hierarchical vs DBSCAN advantages / disadvantages and what each showed on this data.
- **Subtypes**: "PAM50 + claudin-low" and why it is labelled that way.
- **Batch effect**: found in the notebook with PCA coloured by cohort; deliberately not offered in the dashboard.
- **Simplicity decisions**: the controls we removed and why (Phase 5 table).
