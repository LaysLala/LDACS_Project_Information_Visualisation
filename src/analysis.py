"""
Computation layer: statistics, dimensionality reduction, clustering and quality
measures. No plotting here.

Sections
1. Statistics: survival share over time, normalized mutual information.
2. Algorithms: PCA, t-SNE, UMAP, K-means, hierarchical, DBSCAN, purity,
   choosing k (W, B, Calinski-Harabasz), R_NX projection quality.
3. Cached computations for the Clusters & projections page (memoised on the
   filtered patients, so revisiting a setting is instant).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from src.data import load_data


# =============================================================================
# 1. Statistics
# =============================================================================
# Statistics used by the dashboard and the exploratory notebook.
#
# - Survival share over time: for each month t, the share of patients still alive
#   among the patients whose status at t is known. A patient alive at the end of
#   their follow-up only tells us they survived *at least* that long (censoring),
#   so patients whose follow-up ended before t, still alive, are left out at t.
#   A simple proportion per month: accurate over the first years, when almost
#   every patient's status is known; that is why the dashboard shows the first
#   5 years only (later, patients recruited later in the study would be missing
#   from the share and the curve would drop too fast).
# - Normalized mutual information: strength of association between two
#   categorical variables (0 = independent, 1 = one determines the other).
#
# References
# ----------
# Strehl & Ghosh (2002), Cluster ensembles (normalized mutual information). JMLR.

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


# =============================================================================
# 2. Dimensionality reduction, clustering and quality measures
# =============================================================================
# Dimensionality reduction, clustering and their quality measures.
#
# Thin wrappers around scikit-learn / umap-learn with the parameters the user
# can change in the dashboard, plus the evaluation tools seen in the practical
# sessions:
#   - choosing k: within-cluster (W) and between-cluster (B) sums of squares,
#     Calinski-Harabasz index                                  (TP3, lecture 06)
#   - clusters vs known labels: purity                         (TP3 "accuracy")
#   - DBSCAN eps heuristic: sorted k-distance plot              (TP3)
#   - DR quality: Q_NX / R_NX neighbourhood preservation curves (TP6)
#
# The functions are pure (no caching): the dashboard adds caching on top.
#
# References
# ----------
# Pedregosa et al. (2011), Scikit-learn: Machine Learning in Python. JMLR.
# McInnes, Healy & Melville (2018), UMAP. arXiv:1802.03426.
# van der Maaten & Hinton (2008), Visualizing Data using t-SNE. JMLR.
# Ester et al. (1996), A density-based algorithm for discovering clusters (DBSCAN). KDD.
# Calinski & Harabasz (1974), A dendrite method for cluster analysis.
# Lee & Verleysen (2009), Quality assessment of dimensionality reduction:
#     rank-based criteria. Neurocomputing.

SEED = 42


# Dimensionality reduction
def pca(X, n_components: int | None = None) -> tuple[np.ndarray, PCA]:
    """Linear DR. Returns (scores, fitted model); model.explained_variance_ratio_
    gives the scree plot and model.components_ the gene loadings."""
    model = PCA(n_components=n_components, random_state=SEED)
    scores = model.fit_transform(np.asarray(X))
    return scores, model


def tsne(X, perplexity: float = 30.0, seed: int = SEED) -> np.ndarray:
    """Non-linear DR preserving local neighbourhoods (2-D)."""
    return TSNE(
        n_components=2, perplexity=perplexity, init="pca", random_state=seed
    ).fit_transform(np.asarray(X))


def umap(X, n_neighbors: int = 15, min_dist: float = 0.1, seed: int = SEED) -> np.ndarray:
    """Non-linear DR balancing local and global structure (2-D)."""
    import inspect

    import umap as umap_lib  # imported lazily: numba compilation takes a few seconds

    params = dict(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                  random_state=seed, n_jobs=1)  # a fixed seed forces 1 thread anyway
    # umap-learn >= 0.6 announces a new default layout; keep the classic one so
    # results stay identical across versions (older versions lack the option).
    if "compatibility_layout" in inspect.signature(umap_lib.UMAP).parameters:
        params["compatibility_layout"] = True
    return umap_lib.UMAP(**params).fit_transform(np.asarray(X))


# Clustering
def kmeans(X, k: int, seed: int = SEED) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(np.asarray(X))


LINKAGES = ("ward", "centroid", "average", "complete", "single")


def linkage_tree(X, linkage: str = "ward") -> np.ndarray:
    """Full agglomerative merge tree (SciPy linkage matrix Z) for the dendrogram.

    SciPy is used instead of scikit-learn because it offers centroid linkage
    (lecture 06) and returns the whole tree, not only one cut."""
    if linkage not in LINKAGES:
        raise ValueError(f"linkage must be one of {LINKAGES}")
    return hierarchy.linkage(np.asarray(X, dtype=float), method=linkage, metric="euclidean")


def cut_tree(Z: np.ndarray, k: int) -> np.ndarray:
    """Cut a merge tree into (at most) k clusters; labels start at 0."""
    return hierarchy.fcluster(Z, t=k, criterion="maxclust") - 1


def hierarchical(X, k: int, linkage: str = "ward") -> np.ndarray:
    """Agglomerative clustering; linkage in LINKAGES (Ward = default, see report)."""
    return cut_tree(linkage_tree(X, linkage), k)


def count_inversions(Z: np.ndarray) -> int:
    """Merges drawn lower than the previous one. Centroid linkage can produce
    them (lecture 06); a tree with inversions cannot be cut cleanly."""
    return int((np.diff(Z[:, 2]) < 0).sum())


def dbscan(X, eps: float, min_samples: int = 5) -> np.ndarray:
    """Density-based clustering; label -1 = noise."""
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(np.asarray(X))


def k_distance(X, k: int = 5) -> np.ndarray:
    """Distance of every point to its k-th nearest neighbour, sorted ascending.

    The "knee" of this curve is a classic heuristic for DBSCAN's eps.
    scikit-learn's `min_samples` counts the point itself, so the matching
    plot for min_samples = m uses k = m - 1."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(np.asarray(X))
    dist, _ = nn.kneighbors()
    return np.sort(dist[:, k - 1])


def purity(labels, reference) -> float:
    """Share of patients whose cluster's majority reference label is their own.

    1 = every cluster contains a single subtype. Caveat: purity grows with the
    number of clusters (1 cluster per patient gives 1), so compare it at equal
    k and against the one-cluster baseline (share of the largest subtype)."""
    table = pd.crosstab(np.asarray(labels), np.asarray(reference).astype(str))
    return float(table.max(axis=1).sum() / table.to_numpy().sum())


def choose_k_table(X, k_values=range(2, 11), seed: int = SEED) -> pd.DataFrame:
    """K-means quality for several k: W, B and the Calinski-Harabasz index (TP3)."""
    X = np.asarray(X)
    total_ss = float(((X - X.mean(axis=0)) ** 2).sum())
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        w = float(model.inertia_)
        rows.append(
            {
                "k": k,
                "within_ss": w,
                "between_ss": total_ss - w,
                "calinski_harabasz": calinski_harabasz_score(X, model.labels_),
            }
        )
    return pd.DataFrame(rows)


# DR quality: co-ranking matrix, Q_NX(K), R_NX(K)
def _neighbour_ranks(X) -> np.ndarray:
    """R[i, j] = rank of j among i's neighbours (0 = nearest); self gets N-1."""
    D = squareform(pdist(np.asarray(X, dtype=float)))
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1, kind="stable")
    n = len(D)
    ranks = np.empty_like(order)
    ranks[np.arange(n)[:, None], order] = np.arange(n)[None, :]
    return ranks


def rnx_curve(X_high, X_low) -> pd.DataFrame:
    """Neighbourhood preservation for every neighbourhood size K = 1 .. N-2.

    Q_NX(K): average fraction of each point's K nearest neighbours (in the
             original space) that are still among its K nearest neighbours in
             the embedding.
    R_NX(K): Q_NX rescaled so that a random embedding scores 0 and a perfect
             one scores 1, which makes different K comparable.
    """
    n = len(X_high)
    r_high = _neighbour_ranks(X_high)
    r_low = _neighbour_ranks(X_low)
    off_diag = ~np.eye(n, dtype=bool)
    flat = r_high[off_diag] * (n - 1) + r_low[off_diag]
    coranking = np.bincount(flat, minlength=(n - 1) ** 2).reshape(n - 1, n - 1)
    cumulative = coranking.cumsum(axis=0).cumsum(axis=1)

    K = np.arange(1, n - 1)
    q_nx = cumulative[K - 1, K - 1] / (K * n)
    r_nx = ((n - 1) * q_nx - K) / (n - 1 - K)
    return pd.DataFrame({"K": K, "Q_NX": q_nx, "R_NX": r_nx})


def rnx_auc(curve: pd.DataFrame) -> float:
    """Area under R_NX with a log-scale weighting of K (small neighbourhoods
    count more). One number to compare DR methods: higher is better."""
    K = curve["K"].to_numpy()
    return float(np.sum(curve["R_NX"].to_numpy() / K) / np.sum(1.0 / K))


# =============================================================================
# 3. Cached computations for the Clusters & projections page
# =============================================================================
# Cached heavy computations for the Clusters & projections page.
#
# Every function is memoised (functools.lru_cache) on its full set of inputs:
# the method and its parameters and the filtered patients (as a tuple of ids). Re-running the same settings, or coming back to a
# previous one, is instant; only a new combination is computed (VAD rule of
# thumb: responsiveness; brief section 4: limit the cost of interactivity).
#
# Spaces:
#     "genes"  the 489-D mRNA z-scores of the filtered patients
#     "2d"     the current 2-D projection (PCA, t-SNE or UMAP)

MIN_PATIENTS = 20


def ids_key(ids) -> tuple:
    return tuple(sorted(int(i) for i in ids))


@lru_cache(maxsize=8)
def genes(ids: tuple) -> np.ndarray:
    return load_data().mrna.loc[list(ids)].to_numpy()


@lru_cache(maxsize=16)
def projection(method: str, p1: float, p2: float, ids: tuple) -> tuple[np.ndarray, tuple]:
    """2-D coordinates + extra info (explained variance of PC1, PC2 for PCA)."""
    X = genes(ids)
    n = len(ids)
    if method == "pca":
        scores, model = pca(X, 2)
        return scores, tuple(model.explained_variance_ratio_[:2])
    if method == "tsne":
        perplexity = min(p1, max(2.0, (n - 1) / 3))         # t-SNE needs perplexity < n
        return tsne(X, perplexity=perplexity), (perplexity,)
    n_neighbors = int(min(p1, n - 1))
    return umap(X, n_neighbors=n_neighbors, min_dist=p2), (n_neighbors, p2)


def space(name: str, proj: tuple, ids: tuple) -> np.ndarray:
    return genes(ids) if name == "genes" else projection(*proj, ids)[0]


@lru_cache(maxsize=8)
def cached_linkage_tree(linkage: str, space_name: str, proj: tuple, ids: tuple) -> np.ndarray:
    return linkage_tree(space(space_name, proj, ids), linkage)


@lru_cache(maxsize=16)
def cached_k_distance(min_samples: int, space_name: str, proj: tuple, ids: tuple) -> np.ndarray:
    # scikit-learn's min_samples counts the point itself -> k = min_samples - 1
    return k_distance(space(space_name, proj, ids), max(1, min_samples - 1))


@lru_cache(maxsize=16)
def cluster_labels(algo: str, k: int, linkage: str, eps_pct: float, min_samples: int,
                   space_name: str, proj: tuple, ids: tuple) -> tuple[np.ndarray, dict]:
    """Raw labels (DBSCAN noise = -1) + info shown to the user."""
    X = space(space_name, proj, ids)
    if algo == "kmeans":
        return kmeans(X, k), {}
    if algo == "hierarchical":
        Z = cached_linkage_tree(linkage, space_name, proj, ids)
        return cut_tree(Z, k), {"inversions": count_inversions(Z)}
    kd = cached_k_distance(min_samples, space_name, proj, ids)
    eps = float(np.percentile(kd, eps_pct))
    return dbscan(X, eps=eps, min_samples=min_samples), {"eps": eps}


@lru_cache(maxsize=8)
def choose_k(space_name: str, proj: tuple, ids: tuple, k_max: int = 8) -> pd.DataFrame:
    """W, B and Calinski-Harabasz for K-means with k = 2..k_max (3 random starts each, for speed)."""
    X = space(space_name, proj, ids)
    total = float(((X - X.mean(axis=0)) ** 2).sum())
    rows = []
    for k in range(2, k_max + 1):
        model = KMeans(n_clusters=k, n_init=3, random_state=SEED).fit(X)
        rows.append({"k": k, "W": float(model.inertia_), "B": total - float(model.inertia_),
                     "CH": calinski_harabasz_score(X, model.labels_)})
    return pd.DataFrame(rows)


@lru_cache(maxsize=8)
def rnx(proj: tuple, ids: tuple) -> tuple[np.ndarray, np.ndarray, float]:
    """R_NX(K) of the projection w.r.t. the 489-D space, sampled on a log grid of K."""
    curve = rnx_curve(genes(ids), projection(*proj, ids)[0])
    auc = rnx_auc(curve)
    pick = np.unique(np.geomspace(1, len(curve), 150).astype(int)) - 1
    return curve["K"].to_numpy()[pick], curve["R_NX"].to_numpy()[pick], auc


def branch_members(node: int, linkage: str, space_name: str, proj: tuple, ids: tuple) -> list[int]:
    """Patient ids under a node of the merge tree (for clicking a dendrogram branch)."""
    Z = cached_linkage_tree(linkage, space_name, proj, ids)
    nodes = hierarchy.to_tree(Z, rd=True)[1]
    return [ids[i] for i in nodes[node].pre_order()]


def name_clusters(raw: np.ndarray) -> np.ndarray:
    """'C1' = largest cluster, 'C2' = next...; DBSCAN noise -> 'Noise'."""
    raw = np.asarray(raw)
    sizes = pd.Series(raw[raw >= 0]).value_counts()
    names = {lab: f"C{i + 1}" for i, lab in enumerate(sizes.index)}
    return np.array([names.get(r, "Noise") for r in raw], dtype=object)
