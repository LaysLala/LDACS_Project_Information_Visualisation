"""
Figure builders for page 2 - Relationships & genomics.

    association_matrix   normalized mutual information (categorical) or Spearman (numeric)
    pair_figure          cross-table / scatter / box plots, chosen from the two variables' types
    mutation_figure      share of mutated patients: selection vs rest, or per subtype
    expression_heatmap   cluster heatmap: mean expression per group + gene dendrogram

Same conventions as src/figures.py: titles with units, hover on every mark,
selection in colour and the rest in grey, fixed subtype colours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy

from src import stats, theme
from src import variables as V
from src.data import NUMERIC_CLINICAL, UNKNOWN, label, load_data, mrna_matrix
from src.figures import ALL_COLOR, REST_COLOR, SELECTION_COLOR, empty_figure

ASSOC_CATEGORICAL = ["pam50_subtype", "er_status", "pr_status", "her2_status", "three_gene_subtype",
                     "integrative_cluster", "neoplasm_histologic_grade", "tumor_stage", "vital_status"]
SHORT = {
    "pam50_subtype": "PAM50", "er_status": "ER", "pr_status": "PR", "her2_status": "HER2",
    "three_gene_subtype": "3-gene", "integrative_cluster": "Integr. cluster", "neoplasm_histologic_grade": "Grade",
    "tumor_stage": "Stage", "vital_status": "Vital status", "age_at_diagnosis": "Age", "tumor_size": "Tumor size",
    "lymph_nodes_examined_positive": "Pos. nodes", "nottingham_prognostic_index": "NPI",
    "mutation_count": "Mutations", "overall_survival_months": "Follow-up",
}
MIN_PATIENTS = 10


# --------------------------------------------------------------------------- #
# Association matrix
# --------------------------------------------------------------------------- #
def association_values(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, list[str]]:
    """Symmetric matrix of associations between the clinical variables of `mode`."""
    if mode == "num":
        cols = list(NUMERIC_CLINICAL)
        return df[cols].corr(method="spearman", min_periods=MIN_PATIENTS), cols
    cols = ASSOC_CATEGORICAL
    m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            x, y = df[a].astype(str), df[b].astype(str)
            keep = (x != UNKNOWN) & (y != UNKNOWN)          # 'Unknown' is not information
            m.loc[a, b] = m.loc[b, a] = (stats.mutual_information(x[keep], y[keep])
                                         if keep.sum() >= MIN_PATIENTS else np.nan)
    return m, cols


def association_matrix(df: pd.DataFrame, mode: str, picked: tuple[str, str] | None = None,
                       height: int = 330) -> go.Figure:
    if len(df) < MIN_PATIENTS:
        return empty_figure(f"Fewer than {MIN_PATIENTS} patients after filtering", height)
    m, cols = association_values(df, mode)
    rows, columns = cols[1:], cols[:-1]                     # lower triangle without the diagonal
    z = np.full((len(rows), len(columns)), np.nan)
    custom = np.empty((len(rows), len(columns), 2), dtype=object)
    for i, r in enumerate(rows):
        for j, c in enumerate(columns):
            custom[i, j] = [r, c]
            if cols.index(c) < cols.index(r):
                z[i, j] = m.loc[r, c]
    if mode == "num":
        scale, zmin, zmax, title, fmt = theme.DIVERGING, -1, 1, "Spearman ρ", ".2f"
    else:
        top = float(np.nanmax(z)) if np.isfinite(z).any() else 0.35
        scale, zmin, zmax, title, fmt = theme.SEQUENTIAL, 0, max(0.35, top), "Normalized MI", ".2f"
    text = np.where(np.isnan(z), "", np.vectorize(lambda v: format(v, fmt) if np.isfinite(v) else "")(z))

    fig = go.Figure(go.Heatmap(
        z=z, x=[SHORT.get(c, c) for c in columns], y=[SHORT.get(r, r) for r in rows],
        colorscale=scale, zmin=zmin, zmax=zmax, text=text, texttemplate="%{text}",
        textfont=dict(size=11), customdata=custom, xgap=2, ygap=2, hoverongaps=False,
        colorbar=dict(title=dict(text=title, side="right"), thickness=10, len=0.9),
        hovertemplate="%{y} × %{x}<br>" + title + " = %{z:.2f}<br><i>Click to open this pair</i><extra></extra>",
    ))
    if picked:                                   # outline the pair shown in the pair view
        for r, c in (picked, picked[::-1]):      # the cell exists in one orientation only
            if r in rows and c in columns and cols.index(c) < cols.index(r):
                i, j = rows.index(r), columns.index(c)
                fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                              line=dict(color=theme.TEXT, width=2.5), fillcolor="rgba(0,0,0,0)")
                break
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), clickmode="event",
                      xaxis=dict(showgrid=False, tickangle=-35, side="bottom"),
                      yaxis=dict(showgrid=False, autorange="reversed"))
    return fig


# --------------------------------------------------------------------------- #
# Pair view
# --------------------------------------------------------------------------- #
def pair_kind(xkey: str, ykey: str) -> str:
    kinds = {V.describe(xkey).kind, V.describe(ykey).kind}
    return {frozenset({"cat"}): "crosstab", frozenset({"num"}): "scatter"}.get(frozenset(kinds), "box")


def pair_figure(xkey: str, ykey: str, ids, selected: pd.Index, color_key: str | None = None,
                corrected: bool = False, height: int = 330) -> tuple[go.Figure, str]:
    """Figure + one-line summary for any pair of variables."""
    if len(ids) < MIN_PATIENTS:
        return empty_figure(f"Fewer than {MIN_PATIENTS} patients after filtering", height), ""
    if xkey == ykey:
        return empty_figure("Pick two different variables", height), ""
    kind = pair_kind(xkey, ykey)
    if kind == "crosstab":
        return _crosstab(xkey, ykey, ids, selected, corrected, height)
    if kind == "scatter":
        return _scatter(xkey, ykey, ids, selected, color_key, corrected, height)
    numkey, catkey = (xkey, ykey) if V.describe(xkey).kind == "num" else (ykey, xkey)
    return _boxes(numkey, catkey, ids, selected, corrected, height)


def _crosstab(xkey, ykey, ids, selected, corrected, height):
    xs, ys = V.values(xkey, ids, corrected), V.values(ykey, ids, corrected)
    xo, yo = V.categories(xkey, xs), V.categories(ykey, ys)
    counts = pd.crosstab(ys, xs).reindex(index=yo, columns=xo, fill_value=0)
    sel_counts = (pd.crosstab(ys[ys.index.isin(selected)], xs[xs.index.isin(selected)])
                  .reindex(index=yo, columns=xo, fill_value=0))
    share = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).mul(100)
    custom = np.dstack([np.broadcast_to(np.array(xo, dtype=object), counts.shape),
                        np.broadcast_to(np.array(yo, dtype=object)[:, None], counts.shape),
                        counts.to_numpy(), sel_counts.to_numpy()])
    has_sel = len(selected) > 0
    fig = go.Figure(go.Heatmap(
        z=share.to_numpy(), x=xo, y=yo, colorscale=theme.SEQUENTIAL, zmin=0, zmax=100,
        text=share.round(0).astype("Int64").astype(str).to_numpy(), texttemplate="%{text}", textfont=dict(size=11),
        customdata=custom, xgap=2, ygap=2,
        colorbar=dict(title=dict(text="% of row", side="right"), thickness=10, len=0.9),
        hovertemplate=("%{y} → %{x}: %{z:.0f}% of the row (%{customdata[2]:,} patients"
                       + (", %{customdata[3]:,} selected" if has_sel else "") + ")"
                       "<br><i>Click to select these patients</i><extra></extra>"),
    ))
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), clickmode="event",
                      xaxis=dict(title=V.describe(xkey).label, showgrid=False, type="category"),
                      yaxis=dict(title=V.describe(ykey).label, showgrid=False, autorange="reversed", type="category"))
    mi = stats.mutual_information(xs, ys)
    return fig, f"Categorical × categorical → cross-table (row %) · n = {len(ids):,} · normalized mutual information = {mi:.2f}"


def _scatter(xkey, ykey, ids, selected, color_key, corrected, height):
    x, y = V.values(xkey, ids, corrected), V.values(ykey, ids, corrected)
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    has_sel = len(selected) > 0
    fig = go.Figure()
    is_sel = x.index.isin(selected)
    if has_sel:                                   # grey context first, selection on top
        fig.add_trace(go.Scattergl(x=x[~is_sel], y=y[~is_sel], mode="markers", name="Not selected",
                                   marker=dict(color=theme.UNSELECTED_COLOR, size=6, opacity=0.5),
                                   customdata=x.index[~is_sel], hovertemplate="patient %{customdata}<extra></extra>"))
    groups, order, cmap = _colour_groups(color_key, x.index[is_sel] if has_sel else x.index, corrected)
    for g in order:
        m = groups.index[groups == g]
        if len(m) == 0:
            continue
        fig.add_trace(go.Scattergl(
            x=x.loc[m], y=y.loc[m], mode="markers", name=g, marker=dict(color=cmap[g], size=6, opacity=0.8),
            customdata=m, hovertemplate=f"<b>{g}</b><br>patient %{{customdata}}<br>"
                                        f"{V.describe(xkey).short}: %{{x:.2f}}<br>{V.describe(ykey).short}: %{{y:.2f}}<extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), dragmode="select",
                      xaxis_title=V.describe(xkey).label, yaxis_title=V.describe(ykey).label,
                      legend=dict(itemsizing="constant", x=1.01, y=1, xanchor="left"))
    rho = x.corr(y, method="spearman")
    return fig, f"Numeric × numeric → scatter · n = {len(x):,} · Spearman ρ = {rho:.2f} · drag a box to select"


def _colour_groups(color_key, idx, corrected):
    """Colour group per patient for the scatter (categorical colour variables only)."""
    if not color_key:
        return pd.Series("Patients", index=idx), ["Patients"], {"Patients": SELECTION_COLOR}
    vals = V.values(color_key, idx, corrected)
    order = V.categories(color_key, vals)
    name = color_key.split(":", 1)[1]
    if color_key.startswith("mut:"):
        cmap = {V.MUTATED: theme.CATEGORICAL[1], V.NOT_MUTATED: theme.CATEGORICAL[0]}
    else:
        cmap = theme.color_map(name, order)
    return vals, order, cmap


def _boxes(numkey, catkey, ids, selected, corrected, height):
    y, g = V.values(numkey, ids, corrected), V.values(catkey, ids, corrected)
    ok = y.notna()
    y, g = y[ok], g[ok]
    order = V.categories(catkey, g)
    name = catkey.split(":", 1)[1]
    cmap = (theme.color_map(name, order) if catkey.startswith("clin:pam50") or catkey == "clin:vital_status"
            else {c: theme.CATEGORICAL[0] for c in order})
    has_sel = len(selected) > 0
    fig = go.Figure()
    for c in order:
        vals = y[g == c]
        fig.add_trace(go.Box(
            x=[c] * len(vals), y=vals, name=c, marker_color=cmap.get(c, theme.OTHER_COLOR), boxpoints=False,
            showlegend=False, opacity=0.45 if has_sel else 1,
            hovertemplate=f"<b>{c}</b> (n={len(vals):,})<br>%{{y:.2f}}<br><i>Click to select this group</i><extra></extra>"))
    if has_sel:
        s = y.index.isin(selected)
        fig.add_trace(go.Box(
            x=g[s], y=y[s], name=f"Selection (n={int(s.sum()):,})", boxpoints="all", jitter=0.4, pointpos=0,
            marker=dict(color=SELECTION_COLOR, size=5), line=dict(color="rgba(0,0,0,0)"),
            fillcolor="rgba(0,0,0,0)", customdata=y.index[s],
            hovertemplate="selected patient %{customdata}<br>%{y:.2f}<extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), boxmode="overlay", clickmode="event",
                      xaxis=dict(title=V.describe(catkey).label, type="category", categoryorder="array",
                                 categoryarray=order),
                      yaxis_title=V.describe(numkey).label,
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    medians = y.groupby(g).median().reindex(order)
    spread = f"medians from {medians.min():.2f} to {medians.max():.2f}" if len(medians) else ""
    return fig, f"Numeric × categorical → box plots per group · n = {len(y):,} · {spread} · click a box to select its group"


# --------------------------------------------------------------------------- #
# Mutation frequency
# --------------------------------------------------------------------------- #
def top_mutated(ids, n: int) -> list[str]:
    return load_data().mutations.loc[ids].mean().nlargest(n).index.tolist()


def mutation_figure(ids, selected: pd.Index, mode: str, n_genes: int = 8, height: int = 330,
                    clusters: pd.Series | None = None) -> go.Figure:
    if len(ids) < MIN_PATIENTS:
        return empty_figure(f"Fewer than {MIN_PATIENTS} patients after filtering", height)
    muts = load_data().mutations.loc[ids]
    genes = top_mutated(ids, n_genes)
    labels = [g.upper() for g in genes]

    if mode in ("subtype", "cluster"):
        if mode == "cluster" and clusters is not None:
            from src.figures_clusters import NOISE, cluster_order
            sub = clusters.reindex(ids).fillna("Not clustered")
            groups = [c for c in cluster_order(sub[sub != "Not clustered"]) if c != NOISE]
            xtitle = "Cluster (Clusters page)"
        else:
            sub = load_data().clinical.loc[ids, "pam50_subtype"].astype(str)
            groups = [s for s in V.categories("clin:pam50_subtype", sub) if s not in ("NC", UNKNOWN)]
            xtitle = "PAM50 subtype"
        cols = ["All"] + groups
        z = np.array([[muts[g].mean() * 100] + [muts.loc[sub == s, g].mean() * 100 for s in groups] for g in genes])
        n_per = [len(muts)] + [int((sub == s).sum()) for s in groups]
        custom = np.dstack([np.broadcast_to(np.array(genes, dtype=object)[:, None], z.shape),
                            np.broadcast_to(np.array(cols, dtype=object), z.shape),
                            np.broadcast_to(np.array(n_per), z.shape)])
        fig = go.Figure(go.Heatmap(
            z=z, x=cols, y=labels, colorscale=theme.SEQUENTIAL, zmin=0, zmax=max(50, float(np.nanmax(z))),
            text=np.round(z).astype(int).astype(str), texttemplate="%{text}", textfont=dict(size=11),
            customdata=custom, xgap=2, ygap=2,
            colorbar=dict(title=dict(text="% mutated", side="right"), thickness=10, len=0.9),
            hovertemplate="%{y} mutated in %{z:.0f}% of %{x} (n=%{customdata[2]:,})"
                          "<br><i>Click to select these mutated patients</i><extra></extra>"))
        fig.add_vline(x=0.5, line_color=theme.SURFACE, line_width=6)
        fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), clickmode="event",
                          xaxis=dict(showgrid=False, title=xtitle, type="category"),
                          yaxis=dict(showgrid=False, autorange="reversed"))
        return fig

    has_sel = len(selected) > 0
    fig = go.Figure()
    series = ([("Selection", muts.loc[muts.index.isin(selected)], SELECTION_COLOR),
               ("Rest", muts.loc[~muts.index.isin(selected)], REST_COLOR)] if has_sel
              else [("All filtered patients", muts, theme.CATEGORICAL[0])])
    for name, frame, color in series:
        pct = frame[genes].mean().mul(100).to_numpy() if len(frame) else np.zeros(len(genes))
        cnt = frame[genes].sum().to_numpy() if len(frame) else np.zeros(len(genes))
        fig.add_trace(go.Bar(
            y=labels, x=pct, orientation="h", name=f"{name} (n={len(frame):,})", marker_color=color,
            customdata=np.stack([np.array(genes, dtype=object), cnt], axis=1),
            text=[f"{v:.0f}%" for v in pct], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f}% mutated (%{customdata[1]:,} patients)"
                          "<br><i>Click to select the mutated patients</i><extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=8, r=30, t=8, b=8), barmode="group", clickmode="event",
                      bargap=0.25, bargroupgap=0.05,
                      xaxis=dict(title="Patients with ≥ 1 mutation in the gene (%)", range=[0, 100]),
                      yaxis=dict(autorange="reversed", showgrid=False),
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    return fig


# --------------------------------------------------------------------------- #
# Cluster heatmap (mean expression per group + gene dendrogram)
# --------------------------------------------------------------------------- #
def expression_groups(ids, selected: pd.Index, mode: str,
                      clusters: pd.Series | None = None) -> tuple[pd.Series, list[str], dict]:
    if mode == "cluster" and clusters is not None:
        from src.figures_clusters import NOISE, cluster_colors, cluster_order
        g = clusters.reindex(ids).fillna("Not clustered")
        order = [c for c in cluster_order(g[g != "Not clustered"]) if c != NOISE]
        return g, order, cluster_colors(order)
    if mode == "selection" and len(selected):
        g = pd.Series(np.where(pd.Index(ids).isin(selected), "Selection", "Rest"), index=ids)
        return g, ["Selection", "Rest"], {"Selection": SELECTION_COLOR, "Rest": REST_COLOR}
    sub = load_data().clinical.loc[ids, "pam50_subtype"].astype(str)
    order = [s for s in V.categories("clin:pam50_subtype", sub) if s not in ("NC", UNKNOWN)]
    return sub, order, theme.color_map("pam50_subtype", order)


def between_group_share(expr: pd.DataFrame, groups: pd.Series, order: list[str]) -> pd.Series:
    """Per gene: between-group variation B divided by total variation (lecture 06's B, per gene).
    1 = the groups explain all of the gene's variation, 0 = none."""
    keep = groups.isin(order)
    x, g = expr[keep], groups[keep]
    total = ((x - x.mean()) ** 2).sum()
    means, sizes = x.groupby(g).mean(), g.value_counts()
    between = (((means - x.mean()) ** 2).mul(sizes, axis=0)).sum()
    return (between / total).fillna(0)


def expression_heatmap(ids, selected: pd.Index, mode: str, n_genes: int = 20, corrected: bool = False,
                       height: int = 330, clusters: pd.Series | None = None) -> tuple[go.Figure, list[str]]:
    if len(ids) < MIN_PATIENTS:
        return empty_figure(f"Fewer than {MIN_PATIENTS} patients after filtering", height), []
    expr = mrna_matrix(corrected).loc[ids]
    groups, order, cmap = expression_groups(ids, selected, mode, clusters)
    order = [o for o in order if (groups == o).sum() >= 3]
    if len(order) < 2:
        return empty_figure("Need at least two groups of 3 patients", height), []
    share = between_group_share(expr, groups, order)
    genes = share.nlargest(n_genes).index.tolist()
    means = expr[genes].groupby(groups).mean().reindex(order).T          # genes x groups

    Z = hierarchy.linkage(means.to_numpy(), method="ward")               # order genes by similarity
    dendro = hierarchy.dendrogram(Z, no_plot=True, orientation="left")
    leaves = [genes[i] for i in dendro["leaves"]]
    ypos = 5 + 10 * np.arange(len(leaves))
    m = means.loc[leaves]
    lim = float(min(3.0, max(1.0, np.abs(m.to_numpy()).max())))

    fig = make_subplots(rows=1, cols=2, column_widths=[0.14, 0.86], shared_yaxes=True, horizontal_spacing=0.005)
    for xs, ys in zip(dendro["dcoord"], dendro["icoord"]):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=theme.TEXT_SECONDARY, width=1.2),
                                 hoverinfo="skip", showlegend=False), 1, 1)
    custom = np.dstack([np.broadcast_to(np.array(leaves, dtype=object)[:, None], m.shape),
                        np.broadcast_to(np.array(order, dtype=object), m.shape),
                        np.broadcast_to(share.loc[leaves].to_numpy()[:, None], m.shape)])
    fig.add_trace(go.Heatmap(
        z=m.to_numpy(), x=order, y=ypos, colorscale=theme.DIVERGING, zmid=0, zmin=-lim, zmax=lim,
        # values printed in the cells only while rows are tall enough to read them
        text=m.round(1).astype(str).to_numpy(), texttemplate="%{text}" if len(leaves) <= 15 else "",
        textfont=dict(size=10),
        customdata=custom, xgap=2, ygap=1,
        colorbar=dict(title=dict(text="mean z", side="right"), thickness=10, len=0.9),
        hovertemplate=("<b>%{customdata[0]}</b> in %{x}: mean z = %{z:.2f}"
                       "<br>between-group share of variation: %{customdata[2]:.0%}"
                       "<br><i>Click to open this gene in the pair view</i><extra></extra>")), 1, 2)
    fig.update_xaxes(visible=False, autorange="reversed", row=1, col=1)
    fig.update_xaxes(showgrid=False, side="top", row=1, col=2)
    fig.update_yaxes(tickvals=ypos, ticktext=[g.upper() for g in leaves], showgrid=False, row=1, col=1,
                     range=[0, 10 * len(leaves)], tickfont=dict(size=11 if len(leaves) <= 15 else 9))
    fig.update_yaxes(showgrid=False, row=1, col=2)
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=24, b=8), clickmode="event")
    return fig, leaves
