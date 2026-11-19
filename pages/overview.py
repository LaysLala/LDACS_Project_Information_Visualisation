"""
Page 1 - Overview: who are the (filtered) patients and how do they survive?

Cards: KPI tiles, patients per subtype (click = select), Kaplan-Meier survival
split by any variable or by selection vs rest, distribution of a numeric
variable (box-select a range = select), categorical breakdown (click = select).
"""

from __future__ import annotations

import dash
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update

from src import figures
from src.components import card, graph, labelled
from src.data import NUMERIC_CLINICAL, label, load_data
from src.state import categorical_columns, make_selection, visible_selection

dash.register_page(__name__, path="/", name="Overview", title="Overview · METABRIC Explorer", order=0)

SPLIT_OPTIONS = [{"label": label(c), "value": c} for c in categorical_columns()]
BREAKDOWN_OPTIONS = [{"label": "Receptors (ER, PR, HER2)", "value": "receptors"}] + [
    {"label": label(c), "value": c} for c in categorical_columns()
    if c not in ("pam50_subtype", "er_status", "pr_status", "her2_status")]
DIST_OPTIONS = [{"label": label(c), "value": c} for c in NUMERIC_CLINICAL]


def _frames(filtered, selected):
    c = load_data().clinical
    df = c.loc[filtered or []]
    sel = pd.Index(visible_selection(selected, filtered or []))
    return df, sel


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
layout = html.Div([
    html.Div(id="ov-kpis", className="kpi-row", **{"aria-live": "polite"}),

    html.Div([
        card("Patients per PAM50 subtype", [graph("ov-subtypes", 270)],
             help_text="Number of filtered patients in each molecular subtype. With a selection, pale bars "
                       "show all filtered patients and solid bars the selected ones. Click a bar to select "
                       "the patients of that subtype. Colours are the same on every page.",
             class_name="span-5"),
        card("Survival (Kaplan–Meier)", [graph("ov-km", 250), html.P(id="ov-km-test", className="card-note")],
             help_text="Share of patients still alive over time, accounting for patients still alive at the "
                       "end of their follow-up (censoring). Disease-specific: only deaths from breast cancer "
                       "count; overall: deaths of any cause. Curves stop when fewer than 10 patients remain at "
                       "risk. The log-rank test checks whether the curves differ.",
             controls=[
                 labelled("Split by", dcc.Dropdown(id="ov-km-split", options=SPLIT_OPTIONS, value="pam50_subtype",
                                                   clearable=False, className="dropdown")),
                 dcc.RadioItems(id="ov-km-event", className="segmented", inline=True, value="dss_event",
                                options=[{"label": "Disease-specific", "value": "dss_event"},
                                         {"label": "Overall", "value": "os_event"}]),
             ],
             class_name="span-7"),
    ], className="row"),

    html.Div([
        card("Distribution of a clinical variable",
             [html.Div([
                 labelled("Group by", dcc.Dropdown(id="ov-dist-group", options=SPLIT_OPTIONS, value="pam50_subtype",
                                                   clearable=False, className="dropdown")),
                 dcc.Checklist(id="ov-dist-stats", inline=True, className="checks", value=["mean", "median"],
                               options=[{"label": " Mean", "value": "mean"}, {"label": " Median", "value": "median"}]),
                 dcc.Checklist(id="ov-dist-log", inline=True, className="checks", value=[],
                               options=[{"label": " Log scale", "value": "log"}]),
                 labelled("Bins", dcc.Slider(id="ov-dist-bins", min=5, max=60, step=1, value=20,
                                             marks={5: "5", 20: "20", 40: "40", 60: "60"},
                                             tooltip={"placement": "bottom"}), class_name="control slider"),
              ], className="toolbar"),
              graph("ov-dist", 235),
              html.P(id="ov-dist-summary", className="card-note")],
             help_text="Histogram of the chosen variable for the filtered patients (dashed = median, dotted = "
                       "mean). Drag horizontally on the histogram to select the patients in that range. 'Box' "
                       "compares groups (or the selection with the rest). Log scale helps for skewed variables "
                       "such as lymph nodes; the bin count changes what you see.",
             controls=[
                 labelled("Variable", dcc.Dropdown(id="ov-dist-var", options=DIST_OPTIONS, value="age_at_diagnosis",
                                                   clearable=False, className="dropdown wide")),
                 dcc.RadioItems(id="ov-dist-kind", className="segmented", inline=True, value="hist",
                                options=[{"label": "Histogram", "value": "hist"}, {"label": "Box", "value": "box"}]),
             ],
             class_name="span-7"),
        card("Categorical breakdown", [graph("ov-breakdown", 290)],
             help_text="Composition of the filtered patients for one categorical variable, and of the "
                       "selection when there is one. Click a segment to select those patients.",
             controls=[labelled("Variable", dcc.Dropdown(id="ov-break-var", options=BREAKDOWN_OPTIONS,
                                                         value="receptors", clearable=False,
                                                         className="dropdown wide"))],
             class_name="span-5"),
    ], className="row"),
], className="page")


# --------------------------------------------------------------------------- #
# KPI tiles
# --------------------------------------------------------------------------- #
def _fmt(value, kind):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    return {"pct": f"{value:.0%}", "age": f"{value:.1f} y", "months": f"{value:.0f} mo", "n": f"{value:,}"}[kind]


@callback(Output("ov-kpis", "children"), Input("filtered-ids", "data"), Input("selected", "data"))
def kpi_tiles(filtered, selected):
    df, sel = _frames(filtered, selected)
    k = figures.kpis(df)
    ks = figures.kpis(df.loc[sel]) if len(sel) else None
    total = load_data().n_patients
    tiles = [
        ("Patients (after filters)", _fmt(k["n"], "n"), f"of {total:,} in the cohort", "n", "n"),
        ("Median age at diagnosis", _fmt(k["age"], "age"), "years", "age", "age"),
        ("5-year survival (disease)", _fmt(k["dss5"], "pct"), "disease-specific, Kaplan–Meier", "dss5", "pct"),
        ("Died of breast cancer", _fmt(k["died"], "pct"), f"{k.get('died_n', 0):,} patients", "died", "pct"),
        ("Median follow-up time", _fmt(k["followup"], "months"), "months since diagnosis", "followup", "months"),
    ]
    out = []
    for title, value, sub, key, kind in tiles:
        children = [html.Span(title, className="kpi-label"), html.Span(value, className="kpi-value"),
                    html.Span(sub, className="kpi-sub")]
        if ks:
            children.append(html.Span(f"Selection: {_fmt(ks[key], kind)}", className="kpi-sel"))
        out.append(html.Div(children, className="kpi"))
    return out


# --------------------------------------------------------------------------- #
# Subtypes
# --------------------------------------------------------------------------- #
@callback(Output("ov-subtypes", "figure"), Input("filtered-ids", "data"), Input("selected", "data"))
def subtype_chart(filtered, selected):
    df, sel = _frames(filtered, selected)
    return figures.subtype_bars(df, sel)


@callback(Output("selected", "data", allow_duplicate=True), Input("ov-subtypes", "clickData"),
          State("filtered-ids", "data"), prevent_initial_call=True)
def select_subtype(click, filtered):
    if not click or not click.get("points"):
        return no_update
    subtype = click["points"][0]["y"]
    df = load_data().clinical.loc[filtered or []]
    ids = df.index[df["pam50_subtype"].astype(str) == subtype]
    return make_selection(ids, f"subtype {subtype} (Overview)")


# --------------------------------------------------------------------------- #
# Survival
# --------------------------------------------------------------------------- #
@callback(Output("ov-km-split", "options"), Output("ov-km-split", "value"),
          Input("selected", "data"), Input("filtered-ids", "data"), State("ov-km-split", "value"))
def split_options(selected, filtered, current):
    has_sel = bool(visible_selection(selected, filtered or []))
    options = ([{"label": "Selection vs rest", "value": "selection"}] if has_sel else []) + SPLIT_OPTIONS
    if current == "selection" and not has_sel:
        return options, "pam50_subtype"
    return options, no_update


@callback(Output("ov-km", "figure"), Output("ov-km-test", "children"),
          Input("filtered-ids", "data"), Input("selected", "data"),
          Input("ov-km-split", "value"), Input("ov-km-event", "value"))
def survival_chart(filtered, selected, split, event):
    df, sel = _frames(filtered, selected)
    groups, order, colors = figures.survival_groups(df, split, sel)
    return figures.km_figure(df, groups, order, colors, event=event)


# --------------------------------------------------------------------------- #
# Distribution
# --------------------------------------------------------------------------- #
@callback(Output("ov-dist-group", "disabled"), Output("ov-dist-bins", "disabled"), Input("ov-dist-kind", "value"))
def dist_controls(kind):
    return kind == "hist", kind != "hist"


@callback(Output("ov-dist", "figure"), Output("ov-dist-summary", "children"),
          Input("filtered-ids", "data"), Input("selected", "data"), Input("ov-dist-var", "value"),
          Input("ov-dist-kind", "value"), Input("ov-dist-group", "value"), Input("ov-dist-stats", "value"),
          Input("ov-dist-log", "value"), Input("ov-dist-bins", "value"))
def distribution_chart(filtered, selected, column, kind, group_by, stat_flags, log_flag, nbins):
    df, sel = _frames(filtered, selected)
    fig = figures.distribution_figure(df, sel, column, kind=kind, group_by=group_by,
                                      show_mean="mean" in stat_flags, show_median="median" in stat_flags,
                                      log="log" in log_flag, nbins=nbins)
    v = df[column].dropna()
    if len(v) == 0:
        return fig, ""
    missing = len(df) - len(v)
    summary = (f"n = {len(v):,} · mean {v.mean():.1f} · median {v.median():.1f} · "
               f"IQR {v.quantile(.25):.1f}–{v.quantile(.75):.1f} · range {v.min():.1f}–{v.max():.1f}")
    if missing:
        summary += f" · {missing:,} patients without a value are left out"
    return fig, summary


@callback(Output("selected", "data", allow_duplicate=True), Input("ov-dist", "selectedData"),
          State("ov-dist-var", "value"), State("ov-dist-log", "value"), State("ov-dist-kind", "value"),
          State("filtered-ids", "data"), prevent_initial_call=True)
def select_range(selection, column, log_flag, kind, filtered):
    if kind != "hist" or not selection or "range" not in selection:
        return no_update
    lo, hi = selection["range"]["x"]
    if "log" in log_flag:
        lo, hi = 10 ** lo - 1, 10 ** hi - 1
    df = load_data().clinical.loc[filtered or []]
    ids = df.index[df[column].between(lo, hi)]
    return make_selection(ids, f"{label(column)} between {lo:.1f} and {hi:.1f} (Overview)")


# --------------------------------------------------------------------------- #
# Categorical breakdown
# --------------------------------------------------------------------------- #
@callback(Output("ov-breakdown", "figure"), Input("filtered-ids", "data"), Input("selected", "data"),
          Input("ov-break-var", "value"))
def breakdown_chart(filtered, selected, variable):
    df, sel = _frames(filtered, selected)
    return figures.breakdown_figure(df, sel, variable)


@callback(Output("selected", "data", allow_duplicate=True), Input("ov-breakdown", "clickData"),
          State("filtered-ids", "data"), prevent_initial_call=True)
def select_category(click, filtered):
    if not click or not click.get("points"):
        return no_update
    column, category = click["points"][0]["customdata"][:2]
    df = load_data().clinical.loc[filtered or []]
    ids = df.index[df[column].astype(str) == category]
    return make_selection(ids, f"{label(column)} = {category} (Overview)")
