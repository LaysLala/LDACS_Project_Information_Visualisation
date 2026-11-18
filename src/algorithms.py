"""
Dimensionality reduction, clustering and their quality measures.

Thin wrappers around scikit-learn / umap-learn with the parameters the user
can change in the dashboard, plus the evaluation tools seen in the practical
sessions:
  - choosing k: within-cluster (W) and between-cluster (B) sums of squares,
    Calinski-Harabasz index                                  (TP3, lecture 06)
  - clusters vs known labels: purity                         (TP3 "accuracy")
  - DBSCAN eps heuristic: sorted k-distance plot              (TP3)
  - DR quality: Q_NX / R_NX neighbourhood preservation curves (TP6)

The functions are pure (no caching): the dashboard adds caching on top.

References
----------
Pedregosa et al. (2011), Scikit-learn: Machine Learning in Python. JMLR.
McInnes, Healy & Melville (2018), UMAP. arXiv:1802.03426.
van der Maaten & Hinton (2008), Visualizing Data using t-SNE. JMLR.
Ester et al. (1996), A density-based algorithm for discovering clusters (DBSCAN). KDD.
Calinski & Harabasz (1974), A dendrite method for cluster analysis.
Lee & Verleysen (2009), Quality assessment of dimensionality reduction:
    rank-based criteria. Neurocomputing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

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
