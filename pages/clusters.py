"""
Page 3 - Clusters & projections: is there structure in gene expression, and what is it good for?

Settings + Run  -> projection (PCA / t-SNE / UMAP) and clustering (K-means / hierarchical / DBSCAN)
                   on the 489 mRNA z-scores of the filtered patients; cached, stored app-wide
Projection      -> coloured by cluster or any variable; lasso / box = selection
Diagnostics     -> choose k (W, B, CH) or k-distance (DBSCAN) · R_NX quality · dendrogram (click a branch)
Composition     -> cluster x PAM50 subtype + purity (click = selection)
Survival        -> Kaplan-Meier per cluster
"""

from __future__ import annotations

import time

import dash
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update

from src import algorithms as alg
from src import compute as C
from src import figures, figures_clusters as FC
from src import variables as V
from src.components import card, labelled
from src.data import load_data
from src.state import make_selection, visible_selection
from src.theme import CATEGORICAL

dash.register_page(__name__, path="/clusters", name="Clusters & projections",
                   title="Clusters & projections · METABRIC Explorer", order=2)

COLOUR_OPTIONS = [{"label": "Cluster", "value": "cluster"}] + V.options()
METHOD_NAME = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}
ALGO_NAME = {"kmeans": "K-means", "hierarchical": "Hierarchical", "dbscan": "DBSCAN"}
PROJ_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["autoScale2d", "toggleSpikelines"],
               "toImageButtonOptions": {"format": "png", "scale": 2}}
SIMPLE_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}


def _slider(cid, lo, hi, step, value, marks):
    return dcc.Slider(id=cid, min=lo, max=hi, step=step, value=value, marks=marks,
                      tooltip={"placement": "bottom"}, updatemode="mouseup")


def _block(cid, children):
    return html.Div(children, id=cid, className="param-block")


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
settings = card("Settings", [
    html.Div([
        html.Span("Projection (2-D view)", className="param-title"),
        dcc.RadioItems(id="cl-method", className="segmented", inline=True, value="pca", persistence=True,
                       options=[{"label": "PCA", "value": "pca"}, {"label": "t-SNE", "value": "tsne"},
                                {"label": "UMAP", "value": "umap"}]),
        _block("cl-tsne-params", [labelled("Perplexity", _slider("cl-perplexity", 5, 50, 1, 30,
                                                                 {5: "5", 30: "30", 50: "50"}), "control slider")]),
        _block("cl-umap-params", [
            labelled("Neighbours", _slider("cl-neighbors", 5, 100, 1, 15, {5: "5", 15: "15", 50: "50", 100: "100"}),
                     "control slider"),
            labelled("Min. distance", _slider("cl-mindist", 0.0, 0.9, 0.05, 0.1, {0: "0", 0.5: "0.5", 0.9: "0.9"}),
                     "control slider")]),
    ], className="param-group"),
    html.Div([
        html.Span("Clustering", className="param-title"),
        dcc.RadioItems(id="cl-algo", className="segmented", inline=True, value="kmeans", persistence=True,
                       options=[{"label": "K-means", "value": "kmeans"}, {"label": "Hierarch.", "value": "hierarchical"},
                                {"label": "DBSCAN", "value": "dbscan"}]),
        _block("cl-k-params", [labelled("Clusters k", _slider("cl-k", 2, 8, 1, 4, {i: str(i) for i in range(2, 9)}),
                                        "control slider")]),
        _block("cl-link-params", [labelled("Linkage", dcc.Dropdown(
            id="cl-linkage", value="ward", clearable=False, searchable=False, className="dropdown",
            options=[{"label": "Ward (default)", "value": "ward"}, {"label": "Centroid", "value": "centroid"},
                     {"label": "Average", "value": "average"}, {"label": "Complete", "value": "complete"},
                     {"label": "Single", "value": "single"}]))]),
        _block("cl-dbscan-params", [
            labelled("eps (k-distance percentile)", _slider("cl-eps", 50, 99, 1, 90, {50: "50", 75: "75", 90: "90", 99: "99"}),
                     "control slider"),
            labelled("min_samples", _slider("cl-minsamples", 3, 30, 1, 10, {3: "3", 10: "10", 20: "20", 30: "30"}),
                     "control slider")]),
        html.P(id="cl-space-note", className="muted small"),
    ], className="param-group"),
    html.Button("Run", id="cl-run", className="btn btn-primary btn-run"),
    dcc.Loading(html.P(id="cl-status", className="card-note", **{"aria-live": "polite"}),
                type="dot", delay_show=200, color=CATEGORICAL[0]),
], help_text="Choose a 2-D projection of the 489 gene-expression values and a clustering algorithm, then "
             "press Run. PCA and K-means take about a second; t-SNE, UMAP and hierarchical clustering take "
             "longer the first time (results are kept, so going back to a previous setting is instant). The "
             "computation uses the patients kept by the filters and the cohort-correction setting at the time "
             "of the Run. K-means and hierarchical clustering group the patients on all 489 genes; DBSCAN works "
             "on the 2-D view, because in 489 dimensions all patients are about equally far apart and no "
             "density structure exists. DBSCAN's eps is read on the k-distance curve (Diagnostics): a "
             "percentile of the distances to the k-th neighbour, the usual 'knee' heuristic.",
   class_name="span-3 settings-card")

projection_card = card("Projection", [
    dcc.Loading(dcc.Graph(id="cl-proj", config=PROJ_CONFIG, style={"height": "390px"}),
                type="dot", delay_show=300, color=CATEGORICAL[0]),
    html.P(id="cl-proj-note", className="card-note"),
], help_text="Each dot is a patient, placed so that patients with similar gene expression are close. In PCA "
             "the axes are the two directions of largest variance; in t-SNE and UMAP the axes mean nothing, only "
             "neighbourhoods do. Colour by cluster or by any clinical variable, gene or mutation. Lasso or box "
             "select patients to share the selection with every page.",
   controls=[labelled("Colour", dcc.Dropdown(id="cl-colour", options=COLOUR_OPTIONS, value="cluster",
                                              clearable=False, className="dropdown wide"))],
   class_name="span-5")

diagnostics_card = card("Diagnostics", [
    dcc.Loading(dcc.Graph(id="cl-diag-graph", config=SIMPLE_CONFIG, style={"height": "330px"}),
                type="dot", delay_show=300, color=CATEGORICAL[0]),
    html.P(id="cl-diag-note", className="card-note"),
], help_text="Choose k: K-means within-cluster (W) and between-cluster (B) variation and the Calinski-Harabasz "
             "index for k = 2..8 (a clear elbow in W or a peak in CH suggests a k); for DBSCAN, the sorted "
             "k-distance curve with the chosen eps. Projection quality: R_NX(K), the share of each patient's K "
             "nearest neighbours (in 489-D) kept in the 2-D view (0 = random, 1 = perfect). Dendrogram "
             "(hierarchical only): the dashed line is the cut; click a branch to select its patients.",
   controls=[dcc.RadioItems(id="cl-diag", className="segmented", inline=True, value="k", persistence=True,
                            options=[{"label": "Choose k", "value": "k"},
                                     {"label": "Quality", "value": "quality"},
                                     {"label": "Dendrogram", "value": "tree"}])],
   class_name="span-4")

layout = html.Div([
    html.Div([settings, projection_card, diagnostics_card], className="row"),
    html.Div([
        card("Clusters vs PAM50 subtypes", [
            dcc.Loading(dcc.Graph(id="cl-comp", config=SIMPLE_CONFIG, style={"height": "260px"}),
                        type="dot", delay_show=300, color=CATEGORICAL[0]),
            html.P(id="cl-comp-note", className="card-note"),
        ], help_text="Share of each cluster made of each PAM50 subtype (row %). Purity = share of patients whose "
                     "cluster's most frequent subtype is their own; compare it with the baseline of one cluster "
                     "for everybody. Click 'Whole cluster' or a cell to select those patients.",
           class_name="span-5"),
        card("Survival per cluster", [
            dcc.Loading(dcc.Graph(id="cl-km", config=SIMPLE_CONFIG, style={"height": "260px"}),
                        type="dot", delay_show=300, color=CATEGORICAL[0]),
            html.P(id="cl-km-note", className="card-note"),
        ], help_text="Kaplan-Meier curves per cluster: do clusters found from gene expression alone have "
                     "different prognosis? Same rules as on the Overview page.",
           controls=[dcc.RadioItems(id="cl-km-event", className="segmented", inline=True, value="dss_event",
                                    options=[{"label": "Disease-specific", "value": "dss_event"},
                                             {"label": "Overall", "value": "os_event"}])],
           class_name="span-7"),
    ], className="row"),
], className="page")


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
SHOW, HIDE = {}, {"display": "none"}


def clustering_space(algo: str) -> str:
    """K-means / hierarchical use all 489 genes; DBSCAN the 2-D view (in 489-D every patient is about
    equally far from every other, so there is no density structure: curse of dimensionality)."""
    return "2d" if algo == "dbscan" else "genes"


@callback(Output("cl-tsne-params", "style"), Output("cl-umap-params", "style"), Output("cl-k-params", "style"),
          Output("cl-link-params", "style"), Output("cl-dbscan-params", "style"), Output("cl-space-note", "children"),
          Input("cl-method", "value"), Input("cl-algo", "value"))
def parameter_blocks(method, algo):
    note = ("DBSCAN groups the patients in the 2-D view." if algo == "dbscan"
            else "Groups the patients on all 489 genes.")
    return (SHOW if method == "tsne" else HIDE, SHOW if method == "umap" else HIDE,
            SHOW if algo != "dbscan" else HIDE, SHOW if algo == "hierarchical" else HIDE,
            SHOW if algo == "dbscan" else HIDE, note)


@callback(Output("cluster-result", "data"), Output("cl-status", "children"),
          Input("cl-run", "n_clicks"),
          State("cl-method", "value"), State("cl-perplexity", "value"), State("cl-neighbors", "value"),
          State("cl-mindist", "value"), State("cl-algo", "value"), State("cl-k", "value"),
          State("cl-linkage", "value"), State("cl-eps", "value"), State("cl-minsamples", "value"),
          State("filtered-ids", "data"), State("cohort-correction", "value"),
          State("cluster-result", "data"))
def run(n_clicks, method, perplexity, neighbors, mindist, algo, k, linkage, eps_pct, min_samples,
        filtered, cohort, previous):
    space_name = clustering_space(algo)
    if n_clicks is None and previous:            # coming back to the page: keep the last result
        return no_update, _status(previous, from_cache=True)
    ids = C.ids_key(filtered or [])
    if len(ids) < C.MIN_PATIENTS:
        return no_update, f"At least {C.MIN_PATIENTS} patients are needed; the filters keep {len(ids)}."
    corrected = "on" in (cohort or [])
    proj = (method, float(perplexity if method == "tsne" else neighbors), float(mindist))
    if method == "pca":
        proj = ("pca", 0.0, 0.0)
    t0 = time.perf_counter()
    xy, extra = C.projection(*proj, ids, corrected)
    raw, info = C.cluster_labels(algo, int(k), linkage, float(eps_pct), int(min_samples), space_name, proj, ids, corrected)
    names = C.name_clusters(raw)
    result = {
        "ids": list(ids), "x": xy[:, 0].tolist(), "y": xy[:, 1].tolist(), "labels": names.tolist(),
        "method": method, "extra": [float(e) for e in extra], "proj": list(proj), "algo": algo, "k": int(k),
        "linkage": linkage, "eps_pct": float(eps_pct), "min_samples": int(min_samples), "space": space_name,
        "corrected": corrected, "info": info, "seconds": time.perf_counter() - t0,
    }
    return result, _status(result)


def _status(res, from_cache=False):
    n_clusters = len([c for c in set(res["labels"]) if c != "Noise"])
    noise = res["labels"].count("Noise")
    algo = ALGO_NAME[res["algo"]]
    if res["algo"] == "hierarchical":
        algo += f" ({res['linkage']} linkage)"
    if res["algo"] == "dbscan":
        algo += f" (eps = {res['info']['eps']:.2f}, min_samples = {res['min_samples']})"
    where = "489 genes" if res["space"] == "genes" else "the 2-D view"
    parts = [f"{METHOD_NAME[res['method']]} + {algo} on {where}: {n_clusters} cluster{'s' if n_clusters != 1 else ''}",
             f"{noise:,} noise points" if noise else None,
             f"{len(res['ids']):,} patients", "cohort effect removed" if res["corrected"] else None,
             "last result" if from_cache else f"{res['seconds']:.1f} s"]
    text = " · ".join(p for p in parts if p)
    inv = res["info"].get("inversions")
    if inv:
        text += f" · ⚠ {inv:,} inversions in the tree"
    return text


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
def _patients(res, filtered):
    kept = set(filtered or [])
    return pd.Index([i for i in res["ids"] if i in kept])


@callback(Output("cl-proj", "figure"), Output("cl-proj-note", "children"),
          Input("cluster-result", "data"), Input("filtered-ids", "data"), Input("selected", "data"),
          Input("cl-colour", "value"), Input("cohort-correction", "value"))
def projection(res, filtered, selected, colour, cohort):
    if not res:
        return figures.empty_figure("Press Run to compute a projection", 390), ""
    visible = _patients(res, filtered)
    sel = pd.Index(visible_selection(selected, filtered or []))
    fig = FC.projection_figure(res, visible, sel, colour, res["corrected"], height=390)
    notes = []
    if set(filtered or []) != set(res["ids"]):
        notes.append(f"Filters changed since the last Run: showing the {len(visible):,} computed patients that "
                     "still match. Press Run to recompute on the current patients.")
    if ("on" in (cohort or [])) != res["corrected"]:
        notes.append("The cohort-correction setting changed since the last Run: press Run to apply it.")
    if not notes:
        notes.append("Lasso or box select to share a selection with every page.")
    return fig, " ".join(notes)


@callback(Output("selected", "data", allow_duplicate=True), Input("cl-proj", "selectedData"), prevent_initial_call=True)
def projection_select(box):
    if not box or not box.get("points"):
        return no_update
    ids = []
    for p in box["points"]:
        cd = p.get("customdata")
        if cd is not None:
            ids.append(cd[0] if isinstance(cd, list) else cd)
    return make_selection(ids, "lasso in the projection (Clusters)") if ids else no_update


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
@callback(Output("cl-diag-graph", "figure"), Output("cl-diag-note", "children"),
          Input("cluster-result", "data"), Input("cl-diag", "value"))
def diagnostics(res, tab):
    if not res:
        return figures.empty_figure("Press Run first", 330), ""
    ids, proj, corr, sp = tuple(res["ids"]), tuple(res["proj"]), res["corrected"], res["space"]
    where = "489 genes" if sp == "genes" else "the 2-D view"
    if tab == "quality":
        K, R, auc = C.rnx(proj, ids, corr)
        return (FC.rnx_figure(K, R, auc, res["method"], 330),
                f"How well the 2-D view keeps each patient's neighbours from the 489-D space (AUC = {auc:.3f}; "
                "t-SNE is usually best for small K, PCA for large K).")
    if tab == "tree":
        if res["algo"] != "hierarchical":
            return figures.empty_figure("Choose Hierarchical clustering and press Run to see the dendrogram", 330), ""
        Z = C.linkage_tree(res["linkage"], sp, proj, ids, corr)
        inv = alg.count_inversions(Z)
        return (FC.dendrogram_figure(Z, res["k"], len(ids), inv, 330),
                f"{res['linkage'].capitalize()} linkage on {where}; dashed line = cut into {res['k']} clusters "
                "(k slider + Run). Click a dot to select that branch.")
    if res["algo"] == "dbscan":
        kd = C.k_distance(res["min_samples"], sp, proj, ids, corr)
        return (FC.kdistance_figure(kd, res["info"]["eps"], res["eps_pct"], res["min_samples"], 330),
                f"Distance of each patient to its {res['min_samples'] - 1}-th neighbour in {where}, sorted. "
                "eps is usually taken at the knee of the curve.")
    table = C.choose_k(sp, proj, ids, corr)
    best = int(table.loc[table["CH"].idxmax(), "k"])
    return (FC.choose_k_figure(table, res["k"], 330),
            f"K-means in {where}, k = 2..8 (circle = current k). Calinski–Harabasz is highest at k = {best}.")


@callback(Output("selected", "data", allow_duplicate=True), Input("cl-diag-graph", "clickData"),
          State("cluster-result", "data"), State("cl-diag", "value"), prevent_initial_call=True)
def branch_select(click, res, tab):
    if tab != "tree" or not res or res["algo"] != "hierarchical" or not click or not click.get("points"):
        return no_update
    cd = click["points"][0].get("customdata")
    if not cd:
        return no_update
    members = C.branch_members(int(cd[0]), res["linkage"], res["space"], tuple(res["proj"]),
                               tuple(res["ids"]), res["corrected"])
    return make_selection(members, f"dendrogram branch of {len(members):,} patients (Clusters)")


# --------------------------------------------------------------------------- #
# Composition and survival
# --------------------------------------------------------------------------- #
def _cluster_series(res, filtered):
    s = pd.Series(res["labels"], index=res["ids"])
    return s[s.index.isin(set(filtered or []))]


@callback(Output("cl-comp", "figure"), Output("cl-comp-note", "children"),
          Input("cluster-result", "data"), Input("filtered-ids", "data"))
def composition(res, filtered):
    if not res:
        return figures.empty_figure("Press Run first", 260), ""
    clusters = _cluster_series(res, filtered)
    sub = load_data().clinical.loc[clusters.index, "pam50_subtype"].astype(str)
    keep = (clusters != FC.NOISE) & ~sub.isin(["NC", "Unknown"])
    purity = alg.purity(clusters[keep], sub[keep]) if keep.any() else float("nan")
    baseline = sub[keep].value_counts(normalize=True).max() if keep.any() else float("nan")
    return (FC.composition_figure(clusters, sub, 260),
            f"Purity = {purity:.2f} (one cluster for everybody: {baseline:.2f}) · purity grows with k, "
            "so compare settings at the same k.")


@callback(Output("selected", "data", allow_duplicate=True), Input("cl-comp", "clickData"),
          State("cluster-result", "data"), State("filtered-ids", "data"), prevent_initial_call=True)
def composition_select(click, res, filtered):
    if not res or not click or not click.get("points"):
        return no_update
    cluster, subtype = click["points"][0]["customdata"][:2]
    clusters = _cluster_series(res, filtered)
    ids = clusters.index[clusters == cluster]
    if subtype != "all":
        sub = load_data().clinical.loc[ids, "pam50_subtype"].astype(str)
        ids = ids[(sub == subtype).to_numpy()]
        return make_selection(ids, f"{subtype} patients in cluster {cluster} (Clusters)")
    return make_selection(ids, f"cluster {cluster} (Clusters)")


@callback(Output("cl-km", "figure"), Output("cl-km-note", "children"),
          Input("cluster-result", "data"), Input("filtered-ids", "data"), Input("cl-km-event", "value"))
def survival(res, filtered, event):
    if not res:
        return figures.empty_figure("Press Run first", 260), ""
    clusters = _cluster_series(res, filtered)
    df = load_data().clinical.loc[clusters.index]
    order = [c for c in FC.cluster_order(clusters) if c != FC.NOISE]
    return figures.km_figure(df, clusters, order, FC.cluster_colors(order), event=event, height=260)
