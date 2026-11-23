"""
Figure builders for page 3 - Clusters & projections.

    projection_figure   2-D projection coloured by cluster or any variable (lasso / box = selection)
    choose_k_figure     W, B and Calinski-Harabasz for k = 2..8 (TP3, lecture 06)
    kdistance_figure    sorted k-distance curve with the chosen eps (DBSCAN heuristic, TP3)
    rnx_figure          neighbourhood preservation R_NX(K) of the projection (TP6, DR lecture)
    dendrogram_figure   truncated dendrogram with the cut and clickable branches (lectures 05-06)
    composition_figure  cluster x PAM50 subtype (row %) with purity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy

from src import theme
from src import variables as V
from src.figures import empty_figure

NOISE = "Noise"


def cluster_order(labels) -> list[str]:
    names = sorted({l for l in labels if l != NOISE}, key=lambda s: int(s[1:]))
    return names + ([NOISE] if NOISE in set(labels) else [])


def cluster_colors(order: list[str]) -> dict:
    cmap, slot = {}, 0
    for c in order:
        if c == NOISE:
            cmap[c] = theme.NOISE_COLOR
        else:
            cmap[c] = theme.CATEGORICAL[slot] if slot < theme.MAX_CATEGORIES else theme.OTHER_COLOR
            slot += 1
    return cmap


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
def projection_figure(res: dict, visible: pd.Index, selected: pd.Index, colour: str, corrected: bool,
                      height: int = 360) -> go.Figure:
    pts = pd.DataFrame({"x": res["x"], "y": res["y"], "cluster": res["labels"]}, index=res["ids"])
    pts = pts.loc[pts.index.isin(visible)]
    if len(pts) == 0:
        return empty_figure("No computed patient matches the current filters: press Run", height)
    has_sel = len(selected) > 0
    sel_mask = pts.index.isin(selected)
    fig = go.Figure()
    if has_sel:
        rest = pts[~sel_mask]
        fig.add_trace(go.Scattergl(x=rest["x"], y=rest["y"], mode="markers", name="Not selected",
                                   marker=dict(color=theme.UNSELECTED_COLOR, size=5, opacity=0.45),
                                   customdata=rest.index, hovertemplate="patient %{customdata}<extra></extra>"))
    shown = pts[sel_mask] if has_sel else pts

    if colour == "cluster":
        order = cluster_order(pts["cluster"])
        cmap = cluster_colors(order)
        for c in order:
            part = shown[shown["cluster"] == c]
            if len(part):
                fig.add_trace(_points(part, c, cmap[c], f"<b>{c}</b>"))
    else:
        var = V.describe(colour)
        vals = V.values(colour, shown.index, corrected)
        if var.kind == "cat":
            order = V.categories(colour, vals)
            name = colour.split(":", 1)[1]
            cmap = ({V.MUTATED: theme.CATEGORICAL[1], V.NOT_MUTATED: theme.CATEGORICAL[0]}
                    if colour.startswith("mut:") else theme.color_map(name, order))
            for c in order:
                part = shown[(vals == c).to_numpy()]
                if len(part):
                    fig.add_trace(_points(part, c, cmap.get(c, theme.OTHER_COLOR), f"<b>{c}</b>"))
        else:
            is_gene = colour.startswith("gene:")
            fig.add_trace(go.Scattergl(
                x=shown["x"], y=shown["y"], mode="markers", name=var.short, showlegend=False,
                marker=dict(size=6, opacity=0.85, color=vals.to_numpy(),
                            colorscale=theme.DIVERGING if is_gene else theme.SEQUENTIAL,
                            cmin=theme.ZSCORE_RANGE[0] if is_gene else None,
                            cmax=theme.ZSCORE_RANGE[1] if is_gene else None,
                            colorbar=dict(title=dict(text=var.short, side="right"), thickness=10, len=0.8)),
                customdata=np.stack([shown.index, vals.to_numpy(), shown["cluster"]], axis=1),
                hovertemplate=("patient %{customdata[0]} · %{customdata[2]}<br>" + var.short
                               + ": %{customdata[1]:.2f}<extra></extra>")))

    method = res["method"]
    if method == "pca":
        v1, v2 = res["extra"][:2]
        xt, yt, ticks = f"PC1 ({v1:.1%} of variance)", f"PC2 ({v2:.1%} of variance)", True
    else:
        label = "t-SNE" if method == "tsne" else "UMAP"
        xt, yt, ticks = f"{label} 1", f"{label} 2", False      # axes of t-SNE / UMAP carry no meaning
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), dragmode="lasso",
                      legend=dict(itemsizing="constant", x=1.01, y=1, xanchor="left", title_text=""),
                      xaxis=dict(title=xt, showticklabels=ticks, showgrid=ticks, zeroline=False),
                      yaxis=dict(title=yt, showticklabels=ticks, showgrid=ticks, zeroline=False))
    return fig


def _points(part: pd.DataFrame, name: str, color: str, head: str) -> go.Scattergl:
    return go.Scattergl(x=part["x"], y=part["y"], mode="markers", name=f"{name} ({len(part):,})",
                        marker=dict(color=color, size=6, opacity=0.85),
                        customdata=np.stack([part.index, part["cluster"]], axis=1),
                        hovertemplate=head + "<br>patient %{customdata[0]} · cluster %{customdata[1]}<extra></extra>")


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def choose_k_figure(table: pd.DataFrame, k: int | None, height: int = 300) -> go.Figure:
    panels = [("W", "Within-cluster SS (W) ↓"), ("B", "Between-cluster SS (B) ↑"), ("CH", "Calinski–Harabasz ↑")]
    fig = make_subplots(rows=3, cols=1, subplot_titles=[t for _, t in panels], vertical_spacing=0.13, shared_xaxes=True)
    for i, (col, _) in enumerate(panels, start=1):
        fig.add_trace(go.Scatter(x=table["k"], y=table[col], mode="lines+markers", showlegend=False,
                                 line=dict(color=theme.CATEGORICAL[0], width=2), marker=dict(size=6),
                                 hovertemplate="k = %{x}: %{y:,.1f}<extra></extra>"), i, 1)
        if k in set(table["k"]):
            fig.add_trace(go.Scatter(x=[k], y=table.loc[table["k"] == k, col], mode="markers", showlegend=False,
                                     marker=dict(size=11, color="rgba(0,0,0,0)", line=dict(color=theme.TEXT, width=2)),
                                     hoverinfo="skip"), i, 1)
        fig.update_yaxes(rangemode="tozero", row=i, col=1, tickfont=dict(size=11))
    fig.update_xaxes(title_text="Number of clusters k", dtick=1, row=3, col=1)
    fig.update_annotations(font_size=12)
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=24, b=8))
    return fig


def kdistance_figure(kd: np.ndarray, eps: float, pct: float, min_samples: int, height: int = 300) -> go.Figure:
    fig = go.Figure(go.Scatter(x=np.arange(1, len(kd) + 1), y=kd, mode="lines",
                               line=dict(color=theme.CATEGORICAL[0], width=2),
                               hovertemplate="%{x:,} patients have a distance ≤ %{y:.2f}<extra></extra>"))
    fig.add_hline(y=eps, line_dash="dash", line_color=theme.CATEGORICAL[1],
                  annotation_text=f"eps = {eps:.2f} ({pct:.0f}th percentile)", annotation_position="top left")
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), showlegend=False,
                      xaxis_title="Patients sorted by distance",
                      yaxis_title=f"Distance to the {max(1, min_samples - 1)}-th neighbour")
    return fig


def rnx_figure(K: np.ndarray, R: np.ndarray, auc: float, method: str, height: int = 300) -> go.Figure:
    name = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}[method]
    fig = go.Figure(go.Scatter(x=K, y=R, mode="lines", line=dict(color=theme.CATEGORICAL[0], width=2),
                               name=f"{name} (AUC = {auc:.3f})",
                               hovertemplate="K = %{x:,}: R<sub>NX</sub> = %{y:.3f}<extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), showlegend=True,
                      legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
                      xaxis=dict(type="log", dtick=1, title="Neighbourhood size K (log scale)"),
                      yaxis=dict(title="R<sub>NX</sub>(K)", rangemode="tozero"))
    return fig


def dendrogram_figure(Z: np.ndarray, k: int, n: int, inversions: int, height: int = 300) -> go.Figure:
    d = hierarchy.dendrogram(Z, truncate_mode="lastp", p=30, no_plot=True)
    fig = go.Figure()
    for xs, ys in zip(d["icoord"], d["dcoord"]):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=theme.TEXT_SECONDARY, width=1.3),
                                 hoverinfo="skip", showlegend=False))
    # clickable branches: the top of each U is a merge of Z (matched by height) -> node n + row
    used, bx, by, bnode, bsize = set(), [], [], [], []
    for xs, ys in zip(d["icoord"], d["dcoord"]):
        rows = [r for r in np.flatnonzero(np.isclose(Z[:, 2], ys[1])) if r not in used]
        if not rows:
            continue
        r = int(rows[0]); used.add(r)
        bx.append((xs[1] + xs[2]) / 2); by.append(ys[1]); bnode.append(n + r); bsize.append(int(Z[r, 3]))
    leaf_x = 5 + 10 * np.arange(len(d["leaves"]))
    for i, leaf in enumerate(d["leaves"]):
        bx.append(leaf_x[i]); by.append(0); bnode.append(int(leaf))
        bsize.append(int(Z[leaf - n, 3]) if leaf >= n else 1)
    fig.add_trace(go.Scatter(x=bx, y=by, mode="markers", showlegend=False,
                             marker=dict(size=9, color=theme.CATEGORICAL[0], opacity=0.75),
                             customdata=np.stack([bnode, bsize], axis=1),
                             hovertemplate="branch of %{customdata[1]:,} patients<br><i>Click to select them</i><extra></extra>"))
    if 1 < k <= len(Z):
        heights = np.sort(Z[:, 2])
        cut = (heights[-k] + heights[-k + 1]) / 2 if k >= 2 else heights[-1]
        fig.add_hline(y=cut, line_dash="dash", line_color=theme.CATEGORICAL[1],
                      annotation_text=f"cut → {k} clusters", annotation_position="top right")
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), clickmode="event",
                      xaxis=dict(showticklabels=False, showgrid=False, title="Patients (top 30 branches)"),
                      yaxis=dict(title="Merge distance"))
    if inversions:
        fig.add_annotation(text=f"⚠ {inversions:,} inversions: this tree cannot be cut cleanly", x=0.5, y=1,
                           xref="paper", yref="paper", showarrow=False, yanchor="top",
                           font=dict(color=theme.CATEGORICAL[7], size=12), bgcolor="rgba(255,255,255,0.85)")
    return fig


# --------------------------------------------------------------------------- #
# Clusters vs subtypes
# --------------------------------------------------------------------------- #
def composition_figure(clusters: pd.Series, subtypes: pd.Series, height: int = 260) -> go.Figure:
    order = cluster_order(clusters)
    subs = [s for s in V.categories("clin:pam50_subtype", subtypes) if s not in ("NC", "Unknown")]
    table = pd.crosstab(clusters, subtypes).reindex(index=order, columns=subs, fill_value=0)
    share = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).mul(100)
    sizes = clusters.value_counts().reindex(order).fillna(0).astype(int)
    xcols = ["Whole cluster"] + subs
    # numeric x positions: two heatmap traces on one categorical axis would be placed by index, not by name
    fig = go.Figure()
    fig.add_trace(go.Heatmap(                                   # neutral column: cluster sizes
        z=np.zeros((len(order), 1)), x=[0], y=order, colorscale=[[0, "#efeee9"], [1, "#efeee9"]],
        showscale=False, text=[[f"n = {s:,}"] for s in sizes], texttemplate="%{text}", textfont=dict(size=11),
        customdata=np.array([[[c, "all"]] for c in order], dtype=object), xgap=2, ygap=2,
        hovertemplate="%{y}: %{text} patients<br><i>Click to select the whole cluster</i><extra></extra>"))
    custom = np.dstack([np.broadcast_to(np.array(order, dtype=object)[:, None], share.shape),
                        np.broadcast_to(np.array(subs, dtype=object), share.shape)])
    fig.add_trace(go.Heatmap(
        z=share.to_numpy(), x=list(range(1, len(subs) + 1)), y=order, colorscale=theme.SEQUENTIAL, zmin=0, zmax=100,
        text=share.round(0).astype("Int64").astype(str).to_numpy(), texttemplate="%{text}", textfont=dict(size=11),
        customdata=custom, xgap=2, ygap=2,
        colorbar=dict(title=dict(text="% of cluster", side="right"), thickness=10, len=0.9),
        hovertemplate="%{y}: %{z:.0f}% %{customdata[1]}<br><i>Click to select these patients</i><extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), clickmode="event",
                      xaxis=dict(tickvals=list(range(len(xcols))), ticktext=xcols, showgrid=False,
                                 title="PAM50 subtype", zeroline=False),
                      yaxis=dict(autorange="reversed", showgrid=False, type="category"))
    return fig
