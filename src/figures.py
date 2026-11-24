"""
Visualization layer: colours, the Plotly template and every figure builder.

Figure builders are pure functions (data in, Plotly figure out): they know
nothing about callbacks, so they can be tested without a browser and reused by
several pages and by the notebook.

Sections
1. Theme: colours and the Plotly template.
2. Overview page.
3. Relationships & genomics page.
4. Clusters & projections page.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy

from src import analysis, data
from src.data import NUMERIC_CLINICAL, UNKNOWN, label, load_data, ordered_categories


# =============================================================================
# 1. Theme: colours and the Plotly template
# =============================================================================
# Visual identity shared by every chart in the dashboard.
#
# Why one place: the evaluation grid rewards consistency (fonts, colours and
# styles identical across views) and colour-blind accessibility. Every figure
# uses the "metabric" Plotly template, and every colour *meaning* is defined
# once here.
#
# Colour rules
# ------------
# - One colour = one meaning, everywhere. A PAM50 subtype has the same colour on
#   every page, so users can recognise it without reading the legend twice.
# - Categorical palettes were checked with a colour-vision-deficiency validator
#   (OKLab distance between colours under deuteranopia/protanopia/tritanopia).
#   Colour is never the only cue: legends and hover labels are always present.
# - Diverging scale (z-scores, correlations): blue = low/negative, grey = 0,
#   red = high/positive. Two opposite hues with a neutral midpoint, never a
#   rainbow.
# - Sequential scale (counts, mutual information, frequencies): one hue,
#   light -> dark.
# - Selections: selected marks keep their colour; everything else turns light
#   grey and fades.
#
# Call `register_template()` once (app.py or a notebook) before building figures.

# Typography & surfaces
FONT_FAMILY = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
FONT_SIZE = 13
TITLE_SIZE = 15

SURFACE = "#fcfcfb"         # chart background
PAGE = "#f5f5f3"            # page background behind cards
TEXT = "#0b0b0b"            # primary ink
TEXT_SECONDARY = "#52514e"  # secondary ink (axis titles, captions)
TEXT_MUTED = "#898781"      # tick labels, annotations
GRID = "#e1e0d9"            # hairline gridlines
AXIS = "#c3c2b7"            # axis lines / baselines

# Categorical colours

# PAM50 molecular subtypes - Okabe & Ito (2008) colour-blind-safe palette,
# assigned with meaning: the two luminal subtypes share the blue family,
# the aggressive Basal subtype gets the warmest colour.
# Validated: lightness band, chroma and normal-vision checks pass; the closest
# pair under deuteranopia (Normal vs Her2, dE 7.6) is in the "legal with a
# secondary cue" band -> legends/hover always shown.
SUBTYPE_COLORS: dict[str, str] = {
    "LumA": "#0072B2",
    "LumB": "#56B4E9",
    "Her2": "#CC79A7",
    "Basal": "#D55E00",
    "claudin-low": "#E69F00",
    "Normal": "#009E73",
    "NC": "#999999",
    "Unknown": "#cfcfcf",
}

VITAL_STATUS_COLORS: dict[str, str] = {
    "Living": "#0072B2",
    "Died of disease": "#D55E00",
    "Died of other causes": "#999999",
    "Unknown": "#cfcfcf",
}

# Generic categorical palette (clusters and any other categorical variable).
# Fixed order, never cycled: passes every colour-vision check when used in
# this order. At most 8 groups; extra groups are folded into "Other".
CATEGORICAL: list[str] = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
]
MAX_CATEGORIES = len(CATEGORICAL)
OTHER_COLOR = "#b4b2a9"   # "Other" bucket (> 8 groups)
NOISE_COLOR = "#d9d8d2"   # DBSCAN noise points / unknown values

# Continuous scales
# Diverging: blue (low) <-> neutral grey (0) <-> red (high).
DIVERGING: list[list] = [
    [0.0, "#104281"],
    [0.2, "#2a78d6"],
    [0.4, "#9ec5f4"],
    [0.5, "#f0efec"],
    [0.6, "#f4aaa0"],
    [0.8, "#e34948"],
    [1.0, "#9b1c1c"],
]
# mRNA z-scores reach +20 for a few patients; clip the colour range so those
# outliers don't wash out everything else (values beyond are shown at the end
# colour, the exact value stays in the tooltip).
ZSCORE_RANGE = (-3.0, 3.0)

# Sequential: single blue hue, light -> dark.
SEQUENTIAL: list[list] = [
    [0.0, "#f0f5fc"],
    [0.15, "#cde2fb"],
    [0.3, "#9ec5f4"],
    [0.45, "#6da7ec"],
    [0.6, "#3987e5"],
    [0.75, "#256abf"],
    [0.9, "#184f95"],
    [1.0, "#0d366b"],
]

# Selection (linked brushing)
SELECTED_OPACITY = 0.85
UNSELECTED_COLOR = "#d9d8d2"
UNSELECTED_OPACITY = 0.25
MARKER_SIZE = 6

# Helpers
_FIXED_MAPS = {
    "pam50_subtype": SUBTYPE_COLORS,
    "vital_status": VITAL_STATUS_COLORS,
}


def color_map(column: str, categories: list[str]) -> dict[str, str]:
    """Colour for each category of `column`.

    Subtype and vital status use their fixed semantic colours; any other
    variable takes the generic palette in category order. "Unknown" is always
    grey, and categories beyond the 8th are drawn in the "Other" grey.
    """
    fixed = _FIXED_MAPS.get(column)
    if fixed is not None:
        return {c: fixed.get(c, OTHER_COLOR) for c in categories}

    mapping: dict[str, str] = {}
    slot = 0
    for cat in categories:
        if cat == "Unknown":
            mapping[cat] = NOISE_COLOR
        elif slot < MAX_CATEGORIES:
            mapping[cat] = CATEGORICAL[slot]
            slot += 1
        else:
            mapping[cat] = OTHER_COLOR
    return mapping


# Plotly template
_axis = dict(
    showgrid=True,
    gridcolor=GRID,
    gridwidth=1,
    zeroline=False,
    showline=True,
    linecolor=AXIS,
    ticks="outside",
    tickcolor=AXIS,
    tickfont=dict(color=TEXT_MUTED, size=FONT_SIZE - 1),
    title=dict(font=dict(color=TEXT_SECONDARY, size=FONT_SIZE), standoff=8),
    automargin=True,
)

TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(family=FONT_FAMILY, size=FONT_SIZE, color=TEXT),
        title=dict(font=dict(size=TITLE_SIZE, color=TEXT), x=0, xanchor="left"),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=CATEGORICAL,
        xaxis=_axis,
        yaxis=_axis,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=FONT_SIZE - 1, color=TEXT_SECONDARY),
            title=dict(font=dict(size=FONT_SIZE - 1, color=TEXT_SECONDARY)),
            itemsizing="constant",
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=AXIS,
            font=dict(family=FONT_FAMILY, size=FONT_SIZE - 1, color=TEXT),
        ),
        coloraxis=dict(colorbar=dict(outlinewidth=0, thickness=12)),
        # 'simple_white' sets shape lines to width 0, which hides every
        # reference line (medians, thresholds, dendrogram cut): restore them.
        shapedefaults=dict(line=dict(width=1.5, color=TEXT_SECONDARY), opacity=1,
                           fillcolor=TEXT_MUTED),
        margin=dict(l=56, r=16, t=48, b=48),
        bargap=0.15,
    ),
    data=dict(
        scatter=[go.Scatter(marker=dict(size=MARKER_SIZE, line=dict(width=0)))],
        scattergl=[go.Scattergl(marker=dict(size=MARKER_SIZE, line=dict(width=0)))],
        bar=[go.Bar(marker=dict(line=dict(width=0)))],
        histogram=[go.Histogram(marker=dict(line=dict(width=1, color=SURFACE)))],
    ),
)


def register_template() -> None:
    """Make 'simple_white + metabric' the default template for all figures."""
    pio.templates["metabric"] = TEMPLATE
    pio.templates.default = "simple_white+metabric"


# =============================================================================
# 2. Overview page
# =============================================================================
# Figure builders: pure functions (data in, Plotly figure out), so they can be
# tested without the app and reused across pages.
#
# Conventions (see the theme section above and docs/study_guide.md):
# - every axis has a title with units; every trace has a hover template;
# - a selection is shown in colour and the rest in light grey, or as a
#   separate "Selection" series when comparing;
# - PAM50 subtypes always use their fixed colours.

MIN_GROUP = 10        # groups smaller than this are left out of survival curves
SELECTION_COLOR = CATEGORICAL[0]
REST_COLOR = "#9a998f"
ALL_COLOR = "#c9d9ee"


def empty_figure(message: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(color=TEXT_SECONDARY, size=14))
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
    curve = analysis.survival_share(df["overall_survival_months"], df["dss_event"])
    known = df["dss_event"].dropna()
    return {
        "n": len(df),
        "age": float(df["age_at_diagnosis"].median()),
        "dss5": analysis.survival_at(curve, analysis.HORIZON),
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
    colors = color_map(col, order)
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
# Survival over the first 5 years
# --------------------------------------------------------------------------- #
def survival_groups(df: pd.DataFrame, split: str, selected: pd.Index,
                    clusters: pd.Series | None = None) -> tuple[pd.Series, list, dict]:
    """Group label per patient, group order and colours for a survival split."""
    if split == "cluster" and clusters is not None:
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
    return groups, order, color_map(split, order)


def survival_figure(df: pd.DataFrame, groups: pd.Series, order: list, colors: dict,
                    event: str = "dss_event", height: int = 330) -> tuple[go.Figure, str]:
    """Share of patients alive at each month of the first 5 years, one line per group,
    + a one-line summary of the 5-year values."""
    if len(df) == 0:
        return empty_figure("No patients match the filters", height), ""
    t, e = df["overall_survival_months"], df[event]
    sizes = groups.value_counts()
    shown = [g for g in order if sizes.get(g, 0) >= MIN_GROUP]
    dropped = [g for g in order if 0 < sizes.get(g, 0) < MIN_GROUP]

    fig = go.Figure()
    five_year = []
    for g in shown:
        m = (groups == g).to_numpy()
        curve = analysis.survival_share(t[m], e[m])
        fig.add_trace(go.Scatter(
            x=curve["month"], y=curve["survival"], mode="lines",
            name=f"{g} (n={int(m.sum()):,})", line=dict(color=colors.get(g, OTHER_COLOR), width=2),
            customdata=curve[["alive", "known"]],
            hovertemplate=(f"<b>{g}</b><br>month %{{x}}: %{{y:.1%}} alive"
                           "<br>(%{customdata[0]:,} alive of %{customdata[1]:,} patients with a known status)"
                           "<extra></extra>"),
        ))
        five_year.append(f"{g} {analysis.survival_at(curve, analysis.HORIZON):.0%}")
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=40), hovermode="closest",
        xaxis=dict(title="Months since diagnosis", range=[0, analysis.HORIZON], dtick=12),
        yaxis=dict(title="Patients alive (%)", range=[0, 1.02], tickformat=".0%"),
        legend=dict(orientation="v", x=1.01, y=1, xanchor="left"),
    )
    summary = "Alive at 5 years: " + " · ".join(five_year) if five_year else "No group of 10 patients or more."
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
            marker_color=ALL_COLOR if has_sel else CATEGORICAL[0],
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
        marker_sets = [(values, TEXT, "all")]
        if has_sel and values.index.isin(selected).any():
            marker_sets.append((values[values.index.isin(selected)], CATEGORICAL[0], "selection"))
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
            colors = color_map(group_by, order)
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
        cmap = {"Positive": CATEGORICAL[0], "Negative": "#d9d8d2"}
        legend_cats = ["Positive", "Negative"]
    else:
        cats = ordered_categories(variable, df[variable])
        cats_by_col = {variable: cats}
        cmap = color_map(variable, cats)
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
            x=xs, y=ys, orientation="h", name=cat, marker_color=cmap.get(cat, OTHER_COLOR),
            customdata=cds, text=[f"{v:.0f}%" if v >= 10 else "" for v in xs],   # too narrow below 10%: hover
            textposition="inside", insidetextanchor="middle", textangle=0,
            textfont=dict(color="#ffffff" if cat != "Negative" else TEXT),
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


# =============================================================================
# 3. Relationships & genomics page
# =============================================================================
# Figure builders for page 2 - Relationships & genomics.
#
#     association_matrix   normalized mutual information (categorical) or Spearman (numeric)
#     pair_figure          cross-table / scatter / box plots, chosen from the two variables' types
#     mutation_figure      share of mutated patients: selection vs rest, or per subtype
#     expression_heatmap   cluster heatmap: mean expression per group + gene dendrogram
#
# Same conventions as the Overview section: titles with units, hover on every mark,
# selection in colour and the rest in grey, fixed subtype colours.

ASSOC_CATEGORICAL = ["pam50_subtype", "er_status", "pr_status", "her2_status", "three_gene_subtype",
                     "integrative_cluster", "neoplasm_histologic_grade", "tumor_stage", "vital_status"]
SHORT = {
    "pam50_subtype": "Subtype", "er_status": "ER", "pr_status": "PR", "her2_status": "HER2",
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
            m.loc[a, b] = m.loc[b, a] = (analysis.mutual_information(x[keep], y[keep])
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
        scale, zmin, zmax, title, fmt = DIVERGING, -1, 1, "Spearman ρ", ".2f"
    else:
        top = float(np.nanmax(z)) if np.isfinite(z).any() else 0.35
        scale, zmin, zmax, title, fmt = SEQUENTIAL, 0, max(0.35, top), "Normalized MI", ".2f"
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
                              line=dict(color=TEXT, width=2.5), fillcolor="rgba(0,0,0,0)")
                break
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), clickmode="event",
                      xaxis=dict(showgrid=False, tickangle=-35, side="bottom"),
                      yaxis=dict(showgrid=False, autorange="reversed"))
    return fig


# --------------------------------------------------------------------------- #
# Pair view
# --------------------------------------------------------------------------- #
def pair_kind(xkey: str, ykey: str) -> str:
    kinds = {data.describe(xkey).kind, data.describe(ykey).kind}
    return {frozenset({"cat"}): "crosstab", frozenset({"num"}): "scatter"}.get(frozenset(kinds), "box")


def pair_figure(xkey: str, ykey: str, ids, selected: pd.Index, color_key: str | None = None,
                height: int = 330) -> tuple[go.Figure, str]:
    """Figure + one-line summary for any pair of variables."""
    if len(ids) < MIN_PATIENTS:
        return empty_figure(f"Fewer than {MIN_PATIENTS} patients after filtering", height), ""
    if xkey == ykey:
        return empty_figure("Pick two different variables", height), ""
    kind = pair_kind(xkey, ykey)
    if kind == "crosstab":
        return _crosstab(xkey, ykey, ids, selected, height)
    if kind == "scatter":
        return _scatter(xkey, ykey, ids, selected, color_key, height)
    numkey, catkey = (xkey, ykey) if data.describe(xkey).kind == "num" else (ykey, xkey)
    return _boxes(numkey, catkey, ids, selected, height)


def _crosstab(xkey, ykey, ids, selected, height):
    xs, ys = data.values(xkey, ids), data.values(ykey, ids)
    xo, yo = data.categories(xkey, xs), data.categories(ykey, ys)
    counts = pd.crosstab(ys, xs).reindex(index=yo, columns=xo, fill_value=0)
    sel_counts = (pd.crosstab(ys[ys.index.isin(selected)], xs[xs.index.isin(selected)])
                  .reindex(index=yo, columns=xo, fill_value=0))
    share = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).mul(100)
    custom = np.dstack([np.broadcast_to(np.array(xo, dtype=object), counts.shape),
                        np.broadcast_to(np.array(yo, dtype=object)[:, None], counts.shape),
                        counts.to_numpy(), sel_counts.to_numpy()])
    has_sel = len(selected) > 0
    fig = go.Figure(go.Heatmap(
        z=share.to_numpy(), x=xo, y=yo, colorscale=SEQUENTIAL, zmin=0, zmax=100,
        text=share.round(0).astype("Int64").astype(str).to_numpy(), texttemplate="%{text}", textfont=dict(size=11),
        customdata=custom, xgap=2, ygap=2,
        colorbar=dict(title=dict(text="% of row", side="right"), thickness=10, len=0.9),
        hovertemplate=("%{y} → %{x}: %{z:.0f}% of the row (%{customdata[2]:,} patients"
                       + (", %{customdata[3]:,} selected" if has_sel else "") + ")"
                       "<br><i>Click to select these patients</i><extra></extra>"),
    ))
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), clickmode="event",
                      xaxis=dict(title=data.describe(xkey).label, showgrid=False, type="category"),
                      yaxis=dict(title=data.describe(ykey).label, showgrid=False, autorange="reversed", type="category"))
    mi = analysis.mutual_information(xs, ys)
    return fig, f"Categorical × categorical → cross-table (row %) · n = {len(ids):,} · normalized mutual information = {mi:.2f}"


def _scatter(xkey, ykey, ids, selected, color_key, height):
    x, y = data.values(xkey, ids), data.values(ykey, ids)
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    has_sel = len(selected) > 0
    fig = go.Figure()
    is_sel = x.index.isin(selected)
    if has_sel:                                   # grey context first, selection on top
        fig.add_trace(go.Scattergl(x=x[~is_sel], y=y[~is_sel], mode="markers", name="Not selected",
                                   marker=dict(color=UNSELECTED_COLOR, size=6, opacity=0.5),
                                   customdata=x.index[~is_sel], hovertemplate="patient %{customdata}<extra></extra>"))
    groups, order, cmap = _colour_groups(color_key, x.index[is_sel] if has_sel else x.index)
    for g in order:
        m = groups.index[groups == g]
        if len(m) == 0:
            continue
        fig.add_trace(go.Scattergl(
            x=x.loc[m], y=y.loc[m], mode="markers", name=g, marker=dict(color=cmap[g], size=6, opacity=0.8),
            customdata=m, hovertemplate=f"<b>{g}</b><br>patient %{{customdata}}<br>"
                                        f"{data.describe(xkey).short}: %{{x:.2f}}<br>{data.describe(ykey).short}: %{{y:.2f}}<extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), dragmode="select",
                      xaxis_title=data.describe(xkey).label, yaxis_title=data.describe(ykey).label,
                      legend=dict(itemsizing="constant", x=1.01, y=1, xanchor="left"))
    rho = x.corr(y, method="spearman")
    return fig, f"Numeric × numeric → scatter · n = {len(x):,} · Spearman ρ = {rho:.2f} · drag a box to select"


def _colour_groups(color_key, idx):
    """Colour group per patient for the scatter (categorical colour variables only)."""
    if not color_key:
        return pd.Series("Patients", index=idx), ["Patients"], {"Patients": SELECTION_COLOR}
    vals = data.values(color_key, idx)
    order = data.categories(color_key, vals)
    name = color_key.split(":", 1)[1]
    if color_key.startswith("mut:"):
        cmap = {data.MUTATED: CATEGORICAL[1], data.NOT_MUTATED: CATEGORICAL[0]}
    else:
        cmap = color_map(name, order)
    return vals, order, cmap


def _boxes(numkey, catkey, ids, selected, height):
    y, g = data.values(numkey, ids), data.values(catkey, ids)
    ok = y.notna()
    y, g = y[ok], g[ok]
    order = data.categories(catkey, g)
    name = catkey.split(":", 1)[1]
    cmap = (color_map(name, order) if catkey.startswith("clin:pam50") or catkey == "clin:vital_status"
            else {c: CATEGORICAL[0] for c in order})
    has_sel = len(selected) > 0
    fig = go.Figure()
    for c in order:
        vals = y[g == c]
        fig.add_trace(go.Box(
            x=[c] * len(vals), y=vals, name=c, marker_color=cmap.get(c, OTHER_COLOR), boxpoints=False,
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
                      xaxis=dict(title=data.describe(catkey).label, type="category", categoryorder="array",
                                 categoryarray=order),
                      yaxis_title=data.describe(numkey).label,
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
            sub = clusters.reindex(ids).fillna("Not clustered")
            groups = [c for c in cluster_order(sub[sub != "Not clustered"]) if c != NOISE]
            xtitle = "Cluster (Clusters page)"
        else:
            sub = load_data().clinical.loc[ids, "pam50_subtype"].astype(str)
            groups = [s for s in data.categories("clin:pam50_subtype", sub) if s not in ("NC", UNKNOWN)]
            xtitle = "Molecular subtype"
        cols = ["All"] + groups
        z = np.array([[muts[g].mean() * 100] + [muts.loc[sub == s, g].mean() * 100 for s in groups] for g in genes])
        n_per = [len(muts)] + [int((sub == s).sum()) for s in groups]
        custom = np.dstack([np.broadcast_to(np.array(genes, dtype=object)[:, None], z.shape),
                            np.broadcast_to(np.array(cols, dtype=object), z.shape),
                            np.broadcast_to(np.array(n_per), z.shape)])
        fig = go.Figure(go.Heatmap(
            z=z, x=cols, y=labels, colorscale=SEQUENTIAL, zmin=0, zmax=max(50, float(np.nanmax(z))),
            text=np.round(z).astype(int).astype(str), texttemplate="%{text}", textfont=dict(size=11),
            customdata=custom, xgap=2, ygap=2,
            colorbar=dict(title=dict(text="% mutated", side="right"), thickness=10, len=0.9),
            hovertemplate="%{y} mutated in %{z:.0f}% of %{x} (n=%{customdata[2]:,})"
                          "<br><i>Click to select these mutated patients</i><extra></extra>"))
        fig.add_vline(x=0.5, line_color=SURFACE, line_width=6)
        fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), clickmode="event",
                          xaxis=dict(showgrid=False, title=xtitle, type="category"),
                          yaxis=dict(showgrid=False, autorange="reversed"))
        return fig

    has_sel = len(selected) > 0
    fig = go.Figure()
    series = ([("Selection", muts.loc[muts.index.isin(selected)], SELECTION_COLOR),
               ("Rest", muts.loc[~muts.index.isin(selected)], REST_COLOR)] if has_sel
              else [("All filtered patients", muts, CATEGORICAL[0])])
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
        g = clusters.reindex(ids).fillna("Not clustered")
        order = [c for c in cluster_order(g[g != "Not clustered"]) if c != NOISE]
        return g, order, cluster_colors(order)
    if mode == "selection" and len(selected):
        g = pd.Series(np.where(pd.Index(ids).isin(selected), "Selection", "Rest"), index=ids)
        return g, ["Selection", "Rest"], {"Selection": SELECTION_COLOR, "Rest": REST_COLOR}
    sub = load_data().clinical.loc[ids, "pam50_subtype"].astype(str)
    order = [s for s in data.categories("clin:pam50_subtype", sub) if s not in ("NC", UNKNOWN)]
    return sub, order, color_map("pam50_subtype", order)


def between_group_share(expr: pd.DataFrame, groups: pd.Series, order: list[str]) -> pd.Series:
    """Per gene: between-group variation B divided by total variation (lecture 06's B, per gene).
    1 = the groups explain all of the gene's variation, 0 = none."""
    keep = groups.isin(order)
    x, g = expr[keep], groups[keep]
    total = ((x - x.mean()) ** 2).sum()
    means, sizes = x.groupby(g).mean(), g.value_counts()
    between = (((means - x.mean()) ** 2).mul(sizes, axis=0)).sum()
    return (between / total).fillna(0)


def expression_heatmap(ids, selected: pd.Index, mode: str, n_genes: int = 20,
                       height: int = 330, clusters: pd.Series | None = None) -> tuple[go.Figure, list[str]]:
    if len(ids) < MIN_PATIENTS:
        return empty_figure(f"Fewer than {MIN_PATIENTS} patients after filtering", height), []
    expr = load_data().mrna.loc[ids]
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
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=TEXT_SECONDARY, width=1.2),
                                 hoverinfo="skip", showlegend=False), 1, 1)
    custom = np.dstack([np.broadcast_to(np.array(leaves, dtype=object)[:, None], m.shape),
                        np.broadcast_to(np.array(order, dtype=object), m.shape),
                        np.broadcast_to(share.loc[leaves].to_numpy()[:, None], m.shape)])
    fig.add_trace(go.Heatmap(
        z=m.to_numpy(), x=order, y=ypos, colorscale=DIVERGING, zmid=0, zmin=-lim, zmax=lim,
        # values printed in the cells only while rows are tall enough to read them
        text=m.round(1).astype(str).to_numpy(), texttemplate="%{text}" if len(leaves) <= 15 else "",
        textfont=dict(size=11),
        customdata=custom, xgap=2, ygap=1,
        colorbar=dict(title=dict(text="mean z", side="right"), thickness=10, len=0.9),
        hovertemplate=("<b>%{customdata[0]}</b> in %{x}: mean z = %{z:.2f}"
                       "<br>between-group share of variation: %{customdata[2]:.0%}"
                       "<br><i>Click to open this gene in the pair view</i><extra></extra>")), 1, 2)
    fig.update_xaxes(visible=False, autorange="reversed", row=1, col=1)
    fig.update_xaxes(showgrid=False, side="top", row=1, col=2)
    fig.update_yaxes(tickvals=ypos, ticktext=[g.upper() for g in leaves], showgrid=False, row=1, col=1,
                     range=[0, 10 * len(leaves)], tickfont=dict(size=11))
    fig.update_yaxes(showgrid=False, row=1, col=2)
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=24, b=8), clickmode="event")
    return fig, leaves


# =============================================================================
# 4. Clusters & projections page
# =============================================================================
# Figure builders for page 3 - Clusters & projections.
#
#     projection_figure   2-D projection coloured by cluster or any variable (lasso / box = selection)
#     choose_k_figure     W, B and Calinski-Harabasz for k = 2..8 (TP3, lecture 06)
#     kdistance_figure    sorted k-distance curve with the chosen eps (DBSCAN heuristic, TP3)
#     rnx_figure          neighbourhood preservation R_NX(K) of the projection (TP6, DR lecture)
#     dendrogram_figure   truncated dendrogram with the cut and clickable branches (lectures 05-06)
#     composition_figure  cluster x PAM50 subtype (row %) with purity

NOISE = "Noise"


def cluster_order(labels) -> list[str]:
    names = sorted({l for l in labels if l != NOISE}, key=lambda s: int(s[1:]))
    return names + ([NOISE] if NOISE in set(labels) else [])


def cluster_colors(order: list[str]) -> dict:
    cmap, slot = {}, 0
    for c in order:
        if c == NOISE:
            cmap[c] = NOISE_COLOR
        else:
            cmap[c] = CATEGORICAL[slot] if slot < MAX_CATEGORIES else OTHER_COLOR
            slot += 1
    return cmap


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
def projection_figure(res: dict, visible: pd.Index, selected: pd.Index, colour: str,
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
                                   marker=dict(color=UNSELECTED_COLOR, size=5, opacity=0.45),
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
        var = data.describe(colour)
        vals = data.values(colour, shown.index)
        if var.kind == "cat":
            order = data.categories(colour, vals)
            name = colour.split(":", 1)[1]
            cmap = ({data.MUTATED: CATEGORICAL[1], data.NOT_MUTATED: CATEGORICAL[0]}
                    if colour.startswith("mut:") else color_map(name, order))
            for c in order:
                part = shown[(vals == c).to_numpy()]
                if len(part):
                    fig.add_trace(_points(part, c, cmap.get(c, OTHER_COLOR), f"<b>{c}</b>"))
        else:
            is_gene = colour.startswith("gene:")
            fig.add_trace(go.Scattergl(
                x=shown["x"], y=shown["y"], mode="markers", name=var.short, showlegend=False,
                marker=dict(size=6, opacity=0.85, color=vals.to_numpy(),
                            colorscale=DIVERGING if is_gene else SEQUENTIAL,
                            cmin=ZSCORE_RANGE[0] if is_gene else None,
                            cmax=ZSCORE_RANGE[1] if is_gene else None,
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
                                 line=dict(color=CATEGORICAL[0], width=2), marker=dict(size=6),
                                 hovertemplate="k = %{x}: %{y:,.1f}<extra></extra>"), i, 1)
        if k in set(table["k"]):
            fig.add_trace(go.Scatter(x=[k], y=table.loc[table["k"] == k, col], mode="markers", showlegend=False,
                                     marker=dict(size=11, color="rgba(0,0,0,0)", line=dict(color=TEXT, width=2)),
                                     hoverinfo="skip"), i, 1)
        fig.update_yaxes(rangemode="tozero", row=i, col=1, tickfont=dict(size=11))
    fig.update_xaxes(title_text="Number of clusters k", dtick=1, row=3, col=1)
    fig.update_annotations(font_size=12)
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=24, b=8))
    return fig


def kdistance_figure(kd: np.ndarray, eps: float, pct: float, min_samples: int, height: int = 300) -> go.Figure:
    fig = go.Figure(go.Scatter(x=np.arange(1, len(kd) + 1), y=kd, mode="lines",
                               line=dict(color=CATEGORICAL[0], width=2),
                               hovertemplate="%{x:,} patients have a distance ≤ %{y:.2f}<extra></extra>"))
    fig.add_hline(y=eps, line_dash="dash", line_color=CATEGORICAL[1],
                  annotation_text=f"eps = {eps:.2f} ({pct:.0f}th percentile)", annotation_position="top left")
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), showlegend=False,
                      xaxis_title="Patients sorted by distance",
                      yaxis_title=f"Distance to the {max(1, min_samples - 1)}-th neighbour")
    return fig


def rnx_figure(K: np.ndarray, R: np.ndarray, auc: float, method: str, height: int = 300) -> go.Figure:
    name = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}[method]
    fig = go.Figure(go.Scatter(x=K, y=R, mode="lines", line=dict(color=CATEGORICAL[0], width=2),
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
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=TEXT_SECONDARY, width=1.3),
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
                             marker=dict(size=9, color=CATEGORICAL[0], opacity=0.75),
                             customdata=np.stack([bnode, bsize], axis=1),
                             hovertemplate="branch of %{customdata[1]:,} patients<br><i>Click to select them</i><extra></extra>"))
    if 1 < k <= len(Z):
        heights = np.sort(Z[:, 2])
        cut = (heights[-k] + heights[-k + 1]) / 2 if k >= 2 else heights[-1]
        fig.add_hline(y=cut, line_dash="dash", line_color=CATEGORICAL[1],
                      annotation_text=f"cut → {k} clusters", annotation_position="top right")
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), clickmode="event",
                      xaxis=dict(showticklabels=False, showgrid=False, title="Patients (top 30 branches)"),
                      yaxis=dict(title="Merge distance"))
    if inversions:
        fig.add_annotation(text=f"⚠ {inversions:,} inversions: this tree cannot be cut cleanly", x=0.5, y=1,
                           xref="paper", yref="paper", showarrow=False, yanchor="top",
                           font=dict(color=CATEGORICAL[7], size=12), bgcolor="rgba(255,255,255,0.85)")
    return fig


# --------------------------------------------------------------------------- #
# Clusters vs subtypes
# --------------------------------------------------------------------------- #
def composition_figure(clusters: pd.Series, subtypes: pd.Series, height: int = 260) -> go.Figure:
    order = cluster_order(clusters)
    subs = [s for s in data.categories("clin:pam50_subtype", subtypes) if s not in ("NC", "Unknown")]
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
        z=share.to_numpy(), x=list(range(1, len(subs) + 1)), y=order, colorscale=SEQUENTIAL, zmin=0, zmax=100,
        text=share.round(0).astype("Int64").astype(str).to_numpy(), texttemplate="%{text}", textfont=dict(size=11),
        customdata=custom, xgap=2, ygap=2,
        colorbar=dict(title=dict(text="% of cluster", side="right"), thickness=10, len=0.9),
        hovertemplate="%{y}: %{z:.0f}% %{customdata[1]}<br><i>Click to select these patients</i><extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), clickmode="event",
                      xaxis=dict(tickvals=list(range(len(xcols))), ticktext=xcols, showgrid=False,
                                 title="Molecular subtype", zeroline=False),
                      yaxis=dict(autorange="reversed", showgrid=False, type="category"))
    return fig
