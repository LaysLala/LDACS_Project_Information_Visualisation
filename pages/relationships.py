"""
Page 2 - Relationships & genomics: how are the variables, genes and mutations linked?

Cards (2 x 2):
  association matrix  (normalized mutual information / Spearman)  -- click a cell -> pair view
  pair view           (cross-table / scatter / box, from the variable types) -- click / box-select -> selection
  mutation frequency  (selection vs rest, or per subtype)          -- click -> selection
  cluster heatmap     (mean expression per group + gene dendrogram) -- click a gene -> pair view
"""

from __future__ import annotations

import dash
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update

from src import figures_relationships as FR
from src import variables as V
from src.components import card, graph, labelled
from src.data import load_data
from src.state import categorical_columns, make_selection, visible_selection

dash.register_page(__name__, path="/relationships", name="Relationships & genomics",
                   title="Relationships & genomics · METABRIC Explorer", order=1)

VAR_OPTIONS = V.options()
COLOUR_OPTIONS = ([{"label": "No colour", "value": "none"}]
                  + [{"label": V.describe("clin:" + c).label, "value": f"clin:{c}"} for c in categorical_columns()])
DEFAULT_X, DEFAULT_Y = "clin:integrative_cluster", "clin:pam50_subtype"


def _sel(selected, filtered):
    return pd.Index(visible_selection(selected, filtered or []))


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
layout = html.Div([
    html.Div([
        card("How are the variables linked?",
             [graph("rel-matrix", 330), html.P(id="rel-matrix-note", className="card-note")],
             help_text="Lower triangle of an association matrix (the matrix is symmetric, the diagonal is "
                       "always 1). Categorical: normalized mutual information, 0 = independent, 1 = one "
                       "variable determines the other ('Unknown' values are left out). Numeric: Spearman "
                       "rank correlation, -1 to +1, robust to skewed variables. Click a cell to open that "
                       "pair in the pair view.",
             controls=[dcc.RadioItems(id="rel-matrix-mode", className="segmented", inline=True, value="cat",
                                      persistence=True,
                                      options=[{"label": "Categorical · mutual information", "value": "cat"},
                                               {"label": "Numeric · Spearman ρ", "value": "num"}])],
             class_name="span-6"),
        card("Pair view",
             [html.Div([
                 labelled("X", dcc.Dropdown(id="rel-x", options=VAR_OPTIONS, value=DEFAULT_X, clearable=False,
                                            className="dropdown wide")),
                 labelled("Y", dcc.Dropdown(id="rel-y", options=VAR_OPTIONS, value=DEFAULT_Y, clearable=False,
                                            className="dropdown wide")),
                 labelled("Colour", dcc.Dropdown(id="rel-colour", options=COLOUR_OPTIONS, value="clin:pam50_subtype",
                                                 clearable=False, className="dropdown")),
              ], className="toolbar"),
              graph("rel-pair", 300), html.P(id="rel-pair-note", className="card-note")],
             help_text="Any two variables: clinical, gene expression (type a gene name) or mutation status. "
                       "The chart adapts to their types: cross-table (two categorical), scatter (two numeric), "
                       "box plots (one of each). Click a cell or a box, or drag a box on the scatter, to select "
                       "those patients. 'Colour' applies to the scatter.",
             class_name="span-6"),
    ], className="row"),
    html.Div([
        card("Which genes are mutated?",
             [graph("rel-mut", 300), html.P(id="rel-mut-note", className="card-note")],
             help_text="Share of patients with at least one mutation in each of the most frequently mutated "
                       "genes. 'Selection vs rest' compares the current selection with the other filtered "
                       "patients on a common scale; 'By subtype' shows one column per PAM50 subtype. Click a "
                       "bar or a cell to select the mutated patients.",
             controls=[
                 dcc.RadioItems(id="rel-mut-mode", className="segmented", inline=True, value="selection",
                                persistence=True,
                                options=[{"label": "Selection vs rest", "value": "selection"},
                                         {"label": "By subtype", "value": "subtype"},
                                         {"label": "By cluster", "value": "cluster"}]),
                 labelled("Genes", dcc.Dropdown(id="rel-mut-n", options=[8, 12, 16], value=8, clearable=False,
                                                searchable=False, className="mini-dropdown")),
             ],
             class_name="span-6"),
        card("Cluster heatmap: mean expression by group",
             [graph("rel-heat", 300), html.P(id="rel-heat-note", className="card-note")],
             help_text="Mean expression (z-score) of the genes that differ most between the groups, ranked by "
                       "the share of each gene's variation that lies between the groups (between-group "
                       "variation B / total). Rows are reordered by a hierarchical clustering of the genes "
                       "(dendrogram on the left), so genes with similar profiles sit together. Blue = lower "
                       "than average, red = higher. Click a cell to open that gene in the pair view.",
             controls=[
                 dcc.RadioItems(id="rel-heat-mode", className="segmented", inline=True, value="subtype",
                                persistence=True,
                                options=[{"label": "PAM50 subtypes", "value": "subtype"},
                                         {"label": "Clusters", "value": "cluster"},
                                         {"label": "Selection vs rest", "value": "selection"}]),
                 labelled("Genes", dcc.Dropdown(id="rel-heat-n", options=[10, 15, 20], value=15, clearable=False,
                                                searchable=False, className="mini-dropdown")),
             ],
             class_name="span-6"),
    ], className="row"),
], className="page")


# --------------------------------------------------------------------------- #
# Association matrix
# --------------------------------------------------------------------------- #
def _picked(x, y):
    """Matrix cell matching the pair view, if both variables are clinical."""
    if x.startswith("clin:") and y.startswith("clin:"):
        return y.split(":", 1)[1], x.split(":", 1)[1]
    return None


@callback(Output("rel-matrix", "figure"), Output("rel-matrix-note", "children"),
          Input("filtered-ids", "data"), Input("rel-matrix-mode", "value"),
          Input("rel-x", "value"), Input("rel-y", "value"))
def matrix(filtered, mode, x, y):
    df = load_data().clinical.loc[filtered or []]
    fig = FR.association_matrix(df, mode, _picked(x, y))
    what = "normalized mutual information" if mode == "cat" else "Spearman correlation"
    return fig, f"{what.capitalize()} between clinical variables · n = {len(df):,} filtered patients · click a cell"


@callback(Output("rel-x", "value", allow_duplicate=True), Output("rel-y", "value", allow_duplicate=True),
          Input("rel-matrix", "clickData"), prevent_initial_call=True)
def matrix_click(click):
    if not click or not click.get("points"):
        return no_update, no_update
    row, col = click["points"][0]["customdata"]
    return f"clin:{col}", f"clin:{row}"


# --------------------------------------------------------------------------- #
# Pair view
# --------------------------------------------------------------------------- #
@callback(Output("rel-pair", "figure"), Output("rel-pair-note", "children"), Output("rel-colour", "disabled"),
          Input("filtered-ids", "data"), Input("selected", "data"), Input("rel-x", "value"),
          Input("rel-y", "value"), Input("rel-colour", "value"), Input("cohort-correction", "value"))
def pair(filtered, selected, x, y, colour, cohort):
    ids = pd.Index(filtered or [])
    corrected = "on" in (cohort or [])
    fig, note = FR.pair_figure(x, y, ids, _sel(selected, filtered), None if colour == "none" else colour, corrected)
    if corrected and (x.startswith("gene:") or y.startswith("gene:")):
        note += " · cohort effect removed"
    return fig, note, FR.pair_kind(x, y) != "scatter"


@callback(Output("selected", "data", allow_duplicate=True),
          Input("rel-pair", "clickData"), Input("rel-pair", "selectedData"),
          State("rel-x", "value"), State("rel-y", "value"), State("filtered-ids", "data"),
          State("cohort-correction", "value"), prevent_initial_call=True)
def pair_select(click, box, x, y, filtered, cohort):
    trigger = dash.ctx.triggered_id and dash.ctx.triggered[0]["prop_id"].split(".")[1]
    ids = pd.Index(filtered or [])
    corrected = "on" in (cohort or [])
    kind = FR.pair_kind(x, y)
    if trigger == "selectedData" and kind == "scatter":
        if not box or not box.get("points"):
            return no_update
        chosen = [p["customdata"] for p in box["points"] if "customdata" in p]
        return make_selection(chosen, f"box in {V.describe(x).short} × {V.describe(y).short} (Relationships)")
    if trigger != "clickData" or not click or not click.get("points"):
        return no_update
    point = click["points"][0]
    if kind == "crosstab":
        xcat, ycat = point["customdata"][:2]
        xs, ys = V.values(x, ids, corrected), V.values(y, ids, corrected)
        chosen = ids[(xs == xcat).to_numpy() & (ys == ycat).to_numpy()]
        return make_selection(chosen, f"{V.describe(y).short} = {ycat} and {V.describe(x).short} = {xcat} (Relationships)")
    if kind == "box":
        catkey = y if V.describe(x).kind == "num" else x
        category = point.get("x")
        if category is None:
            return no_update
        vals = V.values(catkey, ids, corrected)
        return make_selection(ids[(vals == category).to_numpy()], f"{V.describe(catkey).short} = {category} (Relationships)")
    return no_update


# --------------------------------------------------------------------------- #
# Mutation frequency
# --------------------------------------------------------------------------- #
def _clusters(res):
    return pd.Series(res["labels"], index=res["ids"]) if res else None


NO_CLUSTERS = "No clustering yet: run one on the Clusters & projections page. Showing PAM50 subtypes meanwhile."


@callback(Output("rel-mut", "figure"), Output("rel-mut-note", "children"),
          Input("filtered-ids", "data"), Input("selected", "data"), Input("rel-mut-mode", "value"),
          Input("rel-mut-n", "value"), Input("cluster-result", "data"))
def mutations(filtered, selected, mode, n, res):
    ids = pd.Index(filtered or [])
    sel = _sel(selected, filtered)
    if mode == "cluster" and not res:
        return FR.mutation_figure(ids, sel, "subtype", n), NO_CLUSTERS
    fig = FR.mutation_figure(ids, sel, mode, n, clusters=_clusters(res))
    if mode == "selection" and not len(sel):
        note = "No selection: showing all filtered patients. Select patients anywhere to compare them with the rest."
    else:
        note = f"{n} most frequently mutated genes among the {len(ids):,} filtered patients · click to select"
    return fig, note


@callback(Output("selected", "data", allow_duplicate=True), Input("rel-mut", "clickData"),
          State("rel-mut-mode", "value"), State("filtered-ids", "data"), State("cluster-result", "data"),
          prevent_initial_call=True)
def mutation_select(click, mode, filtered, res):
    if not click or not click.get("points"):
        return no_update
    custom = click["points"][0]["customdata"]
    gene = custom[0]
    d = load_data()
    ids = pd.Index(filtered or [])
    mask = d.mutations.loc[ids, gene] == 1
    source = f"{gene.upper()} mutated"
    if mode in ("subtype", "cluster") and len(custom) > 1 and custom[1] != "All":
        if mode == "cluster" and res:
            groups = _clusters(res).reindex(ids)
            where = f"cluster {custom[1]}"
        else:
            groups = d.clinical.loc[ids, "pam50_subtype"].astype(str)
            where = custom[1]
        mask &= (groups == custom[1]).to_numpy()
        source += f" in {where}"
    return make_selection(ids[mask.to_numpy()], source + " (Relationships)")


# --------------------------------------------------------------------------- #
# Cluster heatmap
# --------------------------------------------------------------------------- #
@callback(Output("rel-heat", "figure"), Output("rel-heat-note", "children"),
          Input("filtered-ids", "data"), Input("selected", "data"), Input("rel-heat-mode", "value"),
          Input("rel-heat-n", "value"), Input("cohort-correction", "value"), Input("cluster-result", "data"))
def heatmap(filtered, selected, mode, n, cohort, res):
    ids = pd.Index(filtered or [])
    sel = _sel(selected, filtered)
    corrected = "on" in (cohort or [])
    if mode == "cluster" and not res:
        fig, genes = FR.expression_heatmap(ids, sel, "subtype", n, corrected)
        return fig, NO_CLUSTERS
    fig, genes = FR.expression_heatmap(ids, sel, mode, n, corrected, clusters=_clusters(res))
    if mode == "selection" and not len(sel):
        note = "No selection: showing PAM50 subtypes. Select patients anywhere to compare them with the rest."
    else:
        note = f"{len(genes)} genes that differ most between the groups · rows ordered by a dendrogram (Ward)"
    if corrected:
        note += " · cohort effect removed"
    return fig, note


@callback(Output("rel-x", "value", allow_duplicate=True), Output("rel-y", "value", allow_duplicate=True),
          Input("rel-heat", "clickData"), prevent_initial_call=True)
def heatmap_click(click):
    if not click or not click.get("points"):
        return no_update, no_update
    custom = click["points"][0].get("customdata")
    if not custom:          # click on the dendrogram
        return no_update, no_update
    return f"gene:{custom[0]}", "clin:pam50_subtype"
