"""
Figure builders: pure functions (data in, Plotly figure out), so they can be
tested without the app and reused across pages.

Conventions (see src/theme.py and docs/study_guide.md):
- every axis has a title with units; every trace has a hover template;
- a selection is shown in colour and the rest in light grey, or as a
  separate "Selection" series when comparing;
- PAM50 subtypes always use their fixed colours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src import stats, theme
from src.data import label
from src.state import ordered_categories

MIN_AT_RISK = 10      # a survival curve stops when fewer patients remain at risk
MIN_GROUP = 10        # groups smaller than this are left out of survival curves
SELECTION_COLOR = theme.CATEGORICAL[0]
REST_COLOR = "#9a998f"
ALL_COLOR = "#c9d9ee"


def empty_figure(message: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(color=theme.TEXT_SECONDARY, size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=height, margin=dict(l=16, r=16, t=16, b=16))
    return fig


# --------------------------------------------------------------------------- #
# KPI tiles
# --------------------------------------------------------------------------- #
def kpis(df: pd.DataFrame) -> dict:
    """Headline numbers for a group of patients."""
    if len(df) == 0:
        return {"n": 0, "age": None, "dss5": None, "died": None, "followup": None}
    curve = stats.kaplan_meier(df["overall_survival_months"], df["dss_event"])
    known = df["dss_event"].dropna()
    return {
        "n": len(df),
        "age": float(df["age_at_diagnosis"].median()),
        "dss5": stats.survival_at(curve, 60),
        "died": float(known.mean()) if len(known) else None,
        "died_n": int(known.sum()),
        "followup": float(df["overall_survival_months"].median()),
    }


# --------------------------------------------------------------------------- #
# Patients per subtype
# --------------------------------------------------------------------------- #
def subtype_bars(df: pd.DataFrame, selected: pd.Index, height: int = 300) -> go.Figure:
    if len(df) == 0:
        return empty_figure("No patients match the filters", height)
    col = "pam50_subtype"
    order = ordered_categories(col, df[col])
    colors = theme.color_map(col, order)
    counts = df[col].astype(str).value_counts().reindex(order, fill_value=0)
    has_sel = len(selected) > 0
    sel_counts = (df.loc[df.index.isin(selected), col].astype(str).value_counts()
                  .reindex(order, fill_value=0))
    total = counts.sum()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=order, x=counts.values, orientation="h", name="All filtered patients",
        marker=dict(color=[colors[c] for c in order], opacity=0.35 if has_sel else 1.0),
        customdata=np.stack([counts.values / total * 100, sel_counts.values], axis=1),
        text=[f"{n:,}" for n in counts.values], textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:,} patients (%{customdata[0]:.1f}%)"
                      + ("<br>%{customdata[1]:,} selected" if has_sel else "")
                      + "<br><i>Click to select this subtype</i><extra></extra>",
    ))
    if has_sel:
        fig.add_trace(go.Bar(
            y=order, x=sel_counts.values, orientation="h", name="Selected",
            marker=dict(color=[colors[c] for c in order]), hoverinfo="skip",
        ))
    fig.update_layout(
        barmode="overlay", height=height, showlegend=False, margin=dict(l=8, r=40, t=8, b=40),
        xaxis_title="Patients", yaxis=dict(autorange="reversed", showgrid=False),
        clickmode="event",
    )
    return fig


# --------------------------------------------------------------------------- #
# Kaplan-Meier
# --------------------------------------------------------------------------- #
def survival_groups(df: pd.DataFrame, split: str, selected: pd.Index,
                    clusters: pd.Series | None = None) -> tuple[pd.Series, list, dict]:
    """Group label per patient, group order and colours for a survival split."""
    if split == "cluster" and clusters is not None:
        from src.figures_clusters import NOISE, cluster_colors, cluster_order
        groups = clusters.reindex(df.index).fillna("Not clustered")
        order = [c for c in cluster_order(clusters[clusters.index.isin(df.index)]) if c != NOISE]
        return groups, order, cluster_colors(order)
    if split == "selection":
        groups = pd.Series(np.where(df.index.isin(selected), "Selection", "Rest"), index=df.index)
        return groups, ["Selection", "Rest"], {"Selection": SELECTION_COLOR, "Rest": REST_COLOR}
    if split == "all":
        groups = pd.Series("All filtered patients", index=df.index)
        return groups, ["All filtered patients"], {"All filtered patients": SELECTION_COLOR}
    groups = df[split].astype(str)
    order = [g for g in ordered_categories(split, groups) if g != "Unknown"]
    return groups, order, theme.color_map(split, order)


def km_figure(df: pd.DataFrame, groups: pd.Series, order: list, colors: dict,
              event: str = "dss_event", height: int = 330) -> tuple[go.Figure, str]:
    """Kaplan-Meier curves per group + a one-line log-rank summary."""
    if len(df) == 0:
        return empty_figure("No patients match the filters", height), ""
    t, e = df["overall_survival_months"], df[event]
    sizes = groups.value_counts()
    shown = [g for g in order if sizes.get(g, 0) >= MIN_GROUP]
    dropped = [g for g in order if 0 < sizes.get(g, 0) < MIN_GROUP]

    fig = go.Figure()
    for g in shown:
        m = (groups == g).to_numpy()
        curve = stats.kaplan_meier(t[m], e[m])
        curve = curve[curve["at_risk"] >= MIN_AT_RISK]
        fig.add_trace(go.Scatter(
            x=curve["time"], y=curve["survival"], mode="lines", line_shape="hv",
            name=f"{g} (n={int(m.sum()):,})", line=dict(color=colors.get(g, theme.OTHER_COLOR), width=2),
            customdata=curve[["at_risk"]],
            hovertemplate=(f"<b>{g}</b><br>%{{x:.0f}} months: %{{y:.1%}} survival"
                           "<br>%{customdata[0]:,} patients still at risk<extra></extra>"),
        ))
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=40), hovermode="closest",
        xaxis_title="Months since diagnosis",
        yaxis=dict(title="Survival probability", range=[0, 1.02], tickformat=".0%"),
        legend=dict(orientation="v", x=1.01, y=1, xanchor="left"),
    )

    if len(shown) >= 2:
        keep = groups.isin(shown).to_numpy()
        chi2, dof, p = stats.logrank_test(t[keep], e[keep], groups[keep])
        p_txt = "< 10⁻³⁰⁰" if p < 1e-300 else (f"= {p:.1e}" if p < 0.001 else f"= {p:.3f}")
        summary = f"Log-rank test: χ² = {chi2:.1f}, df = {dof}, p {p_txt}"
    else:
        summary = "Log-rank test needs at least two groups of 10 patients or more."
    if dropped:
        summary += f" · not shown (< {MIN_GROUP} patients): {', '.join(dropped)}"
    return fig, summary


# --------------------------------------------------------------------------- #
# Distribution of a numeric variable
# --------------------------------------------------------------------------- #
LOG_TICKS = [0, 1, 2, 5, 10, 20, 50, 100, 200]


def _transform(values: pd.Series, log: bool) -> pd.Series:
    return np.log10(1 + values) if log else values


def distribution_figure(df: pd.DataFrame, selected: pd.Index, column: str, kind: str = "hist",
                        group_by: str = "pam50_subtype", show_mean: bool = True,
                        show_median: bool = True, log: bool = False, nbins: int = 20,
                        height: int = 330) -> go.Figure:
    values = df[column].dropna()
    if len(values) == 0:
        return empty_figure("No values to show for these patients", height)
    axis_title = label(column) + (" · log scale" if log else "")
    has_sel = len(selected) > 0
    fig = go.Figure()

    if kind == "hist":
        x = _transform(values, log)
        edges = np.histogram_bin_edges(x, bins=nbins)
        width = edges[1] - edges[0]
        centres = (edges[:-1] + edges[1:]) / 2
        lo_vals = (10 ** edges[:-1] - 1) if log else edges[:-1]
        hi_vals = (10 ** edges[1:] - 1) if log else edges[1:]
        all_counts, _ = np.histogram(x, bins=edges)
        # With a selection, both groups are shown as % of their own size so their SHAPES can be
        # compared (87 patients would be invisible next to 1,904 on a count scale).
        if has_sel:
            sel_vals = _transform(values[values.index.isin(selected)], log)
            sel_counts, _ = np.histogram(sel_vals, bins=edges)
            y_all = all_counts / max(all_counts.sum(), 1) * 100
            y_sel = sel_counts / max(sel_counts.sum(), 1) * 100
        else:
            y_all = all_counts
        fig.add_trace(go.Bar(
            x=centres, y=y_all, width=width * 0.94, name=f"All filtered (n={len(values):,})",
            marker_color=ALL_COLOR if has_sel else theme.CATEGORICAL[0],
            customdata=np.stack([lo_vals, hi_vals, all_counts], axis=1),
            hovertemplate=("%{customdata[0]:.1f} – %{customdata[1]:.1f}: %{customdata[2]:,} patients"
                           + (" (%{y:.1f}% of all filtered)" if has_sel else "") + "<extra></extra>"),
        ))
        if has_sel:
            fig.add_trace(go.Bar(
                x=centres, y=y_sel, width=width * 0.5, name=f"Selection (n={len(sel_vals):,})",
                marker_color=SELECTION_COLOR,
                customdata=np.stack([lo_vals, hi_vals, sel_counts], axis=1),
                hovertemplate=("%{customdata[0]:.1f} – %{customdata[1]:.1f}: %{customdata[2]:,} selected "
                               "(%{y:.1f}% of the selection)<extra></extra>"),
            ))
        marker_sets = [(values, theme.TEXT, "all")]
        if has_sel and values.index.isin(selected).any():
            marker_sets.append((values[values.index.isin(selected)], theme.CATEGORICAL[0], "selection"))
        # mean label on the left of its line, median on the right, selection one line lower:
        # labels never collide even when mean and median are equal
        for row, (vals, color, who) in enumerate(marker_sets):
            for stat, on, dash, side in (("mean", show_mean, "dot", "left"), ("median", show_median, "dash", "right")):
                if not on:
                    continue
                v = vals.mean() if stat == "mean" else vals.median()
                fig.add_vline(x=float(_transform(pd.Series([v]), log).iloc[0]), line_dash=dash,
                              line_color=color, line_width=2)
                fig.add_annotation(x=float(_transform(pd.Series([v]), log).iloc[0]), y=1 - 0.09 * row,
                                   yref="paper", text=f" {stat} {who}: {v:.1f} ", showarrow=False,
                                   xanchor="right" if side == "left" else "left", yanchor="bottom",
                                   font=dict(size=11, color=color), bgcolor="rgba(255,255,255,0.85)")
        fig.update_layout(barmode="overlay", bargap=0, dragmode="select", selectdirection="h",
                          xaxis_title=axis_title,
                          yaxis_title="Patients (% of group)" if has_sel else "Patients")
        if log:
            fig.update_xaxes(tickvals=[np.log10(1 + v) for v in LOG_TICKS if v <= values.max()],
                             ticktext=[str(v) for v in LOG_TICKS if v <= values.max()])
    else:
        sub = df.loc[values.index]
        if has_sel:
            groups = pd.Series(np.where(sub.index.isin(selected), "Selection", "Rest"), index=sub.index)
            order, colors = ["Selection", "Rest"], {"Selection": SELECTION_COLOR, "Rest": REST_COLOR}
        else:
            groups = sub[group_by].astype(str)
            order = ordered_categories(group_by, groups)
            colors = theme.color_map(group_by, order)
        for g in order:
            m = (groups == g).to_numpy()
            if m.sum() == 0:
                continue
            fig.add_trace(go.Box(
                y=_transform(values[m], log), name=f"{g} (n={int(m.sum()):,})", marker_color=colors.get(g),
                boxpoints=False, boxmean=show_mean,
                hovertemplate=f"<b>{g}</b><br>%{{y:.2f}}<extra></extra>",
            ))
        fig.update_layout(showlegend=False, yaxis_title=axis_title,
                          xaxis_title="Selection vs rest" if has_sel else label(group_by))
        if log:
            fig.update_yaxes(tickvals=[np.log10(1 + v) for v in LOG_TICKS if v <= values.max()],
                             ticktext=[str(v) for v in LOG_TICKS if v <= values.max()])

    fig.update_layout(height=height, margin=dict(l=8, r=8, t=40 if has_sel else 24, b=40),
                      legend=dict(orientation="h", y=-0.22, x=1, xanchor="right", yanchor="top"))
    if kind == "hist" and has_sel:          # room for the legend under the x-axis title
        fig.update_layout(margin=dict(b=70))
    return fig


# --------------------------------------------------------------------------- #
# Categorical breakdown (100% stacked bars, all vs selection)
# --------------------------------------------------------------------------- #
RECEPTORS = ["er_status", "pr_status", "her2_status"]


def breakdown_figure(df: pd.DataFrame, selected: pd.Index, variable: str, height: int = 300) -> go.Figure:
    if len(df) == 0:
        return empty_figure("No patients match the filters", height)
    has_sel = len(selected) > 0
    sel_df = df.loc[df.index.isin(selected)]
    rows = []                                     # (row label, column, sub-frame)
    columns = RECEPTORS if variable == "receptors" else [variable]
    for col in columns:
        name = label(col).replace(" status", "") if variable == "receptors" else ""
        rows.append((f"{name} · all (n={len(df):,})".strip(" ·"), col, df))
        if has_sel:
            rows.append((f"{name} · selection (n={len(sel_df):,})".strip(" ·"), col, sel_df))

    fig = go.Figure()
    if variable == "receptors":
        cats_by_col = {c: ["Positive", "Negative"] for c in RECEPTORS}
        cmap = {"Positive": theme.CATEGORICAL[0], "Negative": "#d9d8d2"}
        legend_cats = ["Positive", "Negative"]
    else:
        cats = ordered_categories(variable, df[variable])
        cats_by_col = {variable: cats}
        cmap = theme.color_map(variable, cats)
        legend_cats = cats

    for cat in legend_cats:
        xs, ys, cds = [], [], []
        for row_label, col, sub in rows:
            if cat not in cats_by_col[col]:
                continue
            share = (sub[col].astype(str) == cat).mean() * 100 if len(sub) else 0
            xs.append(share)
            ys.append(row_label)
            cds.append([col, cat, int((sub[col].astype(str) == cat).sum())])
        fig.add_trace(go.Bar(
            x=xs, y=ys, orientation="h", name=cat, marker_color=cmap.get(cat, theme.OTHER_COLOR),
            customdata=cds, text=[f"{v:.0f}%" if v >= 10 else "" for v in xs],   # too narrow below 10%: hover
            textposition="inside", insidetextanchor="middle", textangle=0,
            textfont=dict(color="#ffffff" if cat != "Negative" else theme.TEXT),
            hovertemplate="%{y}<br><b>%{customdata[1]}</b>: %{x:.1f}% (%{customdata[2]:,} patients)"
                          "<br><i>Click to select these patients</i><extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", height=height, margin=dict(l=8, r=8, t=30, b=40),
        xaxis=dict(title="Patients (%)", range=[0, 100]),
        yaxis=dict(autorange="reversed", showgrid=False),
        legend=dict(orientation="h", y=1.0, x=0, yanchor="bottom", title_text="", traceorder="normal"),
        clickmode="event",
    )
    return fig
