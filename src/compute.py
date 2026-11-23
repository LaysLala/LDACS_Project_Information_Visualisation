"""
Cached heavy computations for the Clusters & projections page.

Every function is memoised (functools.lru_cache) on its full set of inputs:
the method and its parameters and the filtered patients (as a tuple of ids). Re-running the same settings, or coming back to a
previous one, is instant; only a new combination is computed (VAD rule of
thumb: responsiveness; brief section 4: limit the cost of interactivity).

Spaces:
    "genes"  the 489-D mRNA z-scores of the filtered patients
    "2d"     the current 2-D projection (PCA, t-SNE or UMAP)
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

from src import algorithms as alg
from src.data import load_data

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
        scores, model = alg.pca(X, 2)
        return scores, tuple(model.explained_variance_ratio_[:2])
    if method == "tsne":
        perplexity = min(p1, max(2.0, (n - 1) / 3))         # t-SNE needs perplexity < n
        return alg.tsne(X, perplexity=perplexity), (perplexity,)
    n_neighbors = int(min(p1, n - 1))
    return alg.umap(X, n_neighbors=n_neighbors, min_dist=p2), (n_neighbors, p2)


def space(name: str, proj: tuple, ids: tuple) -> np.ndarray:
    return genes(ids) if name == "genes" else projection(*proj, ids)[0]


@lru_cache(maxsize=8)
def linkage_tree(linkage: str, space_name: str, proj: tuple, ids: tuple) -> np.ndarray:
    return alg.linkage_tree(space(space_name, proj, ids), linkage)


@lru_cache(maxsize=16)
def k_distance(min_samples: int, space_name: str, proj: tuple, ids: tuple) -> np.ndarray:
    # scikit-learn's min_samples counts the point itself -> k = min_samples - 1
    return alg.k_distance(space(space_name, proj, ids), max(1, min_samples - 1))


@lru_cache(maxsize=16)
def cluster_labels(algo: str, k: int, linkage: str, eps_pct: float, min_samples: int,
                   space_name: str, proj: tuple, ids: tuple) -> tuple[np.ndarray, dict]:
    """Raw labels (DBSCAN noise = -1) + info shown to the user."""
    X = space(space_name, proj, ids)
    if algo == "kmeans":
        return alg.kmeans(X, k), {}
    if algo == "hierarchical":
        Z = linkage_tree(linkage, space_name, proj, ids)
        return alg.cut_tree(Z, k), {"inversions": alg.count_inversions(Z)}
    kd = k_distance(min_samples, space_name, proj, ids)
    eps = float(np.percentile(kd, eps_pct))
    return alg.dbscan(X, eps=eps, min_samples=min_samples), {"eps": eps}


@lru_cache(maxsize=8)
def choose_k(space_name: str, proj: tuple, ids: tuple, k_max: int = 8) -> pd.DataFrame:
    """W, B and Calinski-Harabasz for K-means with k = 2..k_max (3 random starts each, for speed)."""
    X = space(space_name, proj, ids)
    total = float(((X - X.mean(axis=0)) ** 2).sum())
    rows = []
    for k in range(2, k_max + 1):
        model = KMeans(n_clusters=k, n_init=3, random_state=alg.SEED).fit(X)
        rows.append({"k": k, "W": float(model.inertia_), "B": total - float(model.inertia_),
                     "CH": calinski_harabasz_score(X, model.labels_)})
    return pd.DataFrame(rows)


@lru_cache(maxsize=8)
def rnx(proj: tuple, ids: tuple) -> tuple[np.ndarray, np.ndarray, float]:
    """R_NX(K) of the projection w.r.t. the 489-D space, sampled on a log grid of K."""
    curve = alg.rnx_curve(genes(ids), projection(*proj, ids)[0])
    auc = alg.rnx_auc(curve)
    pick = np.unique(np.geomspace(1, len(curve), 150).astype(int)) - 1
    return curve["K"].to_numpy()[pick], curve["R_NX"].to_numpy()[pick], auc


def branch_members(node: int, linkage: str, space_name: str, proj: tuple, ids: tuple) -> list[int]:
    """Patient ids under a node of the merge tree (for clicking a dendrogram branch)."""
    Z = linkage_tree(linkage, space_name, proj, ids)
    nodes = hierarchy.to_tree(Z, rd=True)[1]
    return [ids[i] for i in nodes[node].pre_order()]


def name_clusters(raw: np.ndarray) -> np.ndarray:
    """'C1' = largest cluster, 'C2' = next...; DBSCAN noise -> 'Noise'."""
    raw = np.asarray(raw)
    sizes = pd.Series(raw[raw >= 0]).value_counts()
    names = {lab: f"C{i + 1}" for i, lab in enumerate(sizes.index)}
    return np.array([names.get(r, "Noise") for r in raw], dtype=object)
