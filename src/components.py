"""
Layout building blocks shared by every page: header, filter sidebar,
selection bar, help panel, and the `card` wrapper used for every chart.

Styling lives in assets/style.css; colours and fonts match src/theme.py.
"""

from __future__ import annotations

from dash import dcc, html

from src import theme
from src.data import label
from src.state import CHECKLIST_FILTERS, age_bounds, checklist_options

PAGES = [  # (path, tab label)
    ("/", "Overview"),
    ("/relationships", "Relationships & genomics"),
    ("/clusters", "Clusters & projections"),
]

GRAPH_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "autoScale2d", "toggleSpikelines"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


# --------------------------------------------------------------------------- #
# Generic pieces
# --------------------------------------------------------------------------- #
def info(text: str) -> html.Details:
    """Small (i) button that reveals a help text: details on demand, no callback."""
    return html.Details([html.Summary("i", title="About this view", **{"aria-label": "About this view"}),
                         html.P(text)], className="info")


def card(title, body, help_text: str | None = None, controls=None, card_id: str | None = None,
         class_name: str = "") -> html.Section:
    head = [html.H2(title, className="card-title")]
    if help_text:
        head.append(info(help_text))
    if controls is not None:
        head.append(html.Div(controls, className="card-controls"))
    kwargs = {"id": card_id} if card_id else {}
    return html.Section([html.Div(head, className="card-header"), *body],
                        className=f"card {class_name}".strip(), **kwargs)


def graph(graph_id: str, height: int = 320) -> dcc.Loading:
    """A Plotly graph with the shared config and a loading indicator (shown after 300 ms)."""
    return dcc.Loading(
        dcc.Graph(id=graph_id, config=GRAPH_CONFIG, style={"height": f"{height}px"}),
        type="dot", delay_show=300, color=theme.CATEGORICAL[0],
    )


def labelled(text: str, control, class_name: str = "control") -> html.Label:
    return html.Label([html.Span(text, className="control-label"), control], className=class_name)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
def header() -> html.Header:
    nav = [dcc.Link(text, href=path, id=f"nav-{i}", className="nav-link") for i, (path, text) in enumerate(PAGES)]
    return html.Header([
        html.Div([
            html.Span("●", className="logo-dot", **{"aria-hidden": "true"}),
            html.Span("METABRIC Explorer", className="app-title"),
        ], className="brand"),
        html.Nav(nav, className="nav", **{"aria-label": "Pages"}),
        html.Div([
            html.Span(id="patient-counter", className="counter", **{"aria-live": "polite"}),
            html.Button("Reset filters & selection", id="reset-all", className="btn"),
            html.Button("Help", id="help-open", className="btn btn-primary"),
        ], className="header-actions"),
    ], className="header")


# --------------------------------------------------------------------------- #
# Sidebar (global filters)
# --------------------------------------------------------------------------- #
def _segmented(control_id: str, options: list[tuple[str, str]], value: str) -> dcc.RadioItems:
    return dcc.RadioItems(
        id=control_id, value=value, inline=True, className="segmented",
        options=[{"label": text, "value": v} for v, text in options],
    )


def _chips(control_id: str, column: str) -> dcc.Checklist:
    opts = checklist_options(column)
    return dcc.Checklist(id=control_id, options=[{"label": o, "value": o} for o in opts],
                         value=opts, inline=True, className="chips")


def subtype_option(name: str, count: int | None) -> dict:
    color = theme.SUBTYPE_COLORS.get(name, theme.OTHER_COLOR)
    text = "NC (not classified)" if name == "NC" else name
    return {"value": name, "label": html.Span([
        html.Span(className="swatch", style={"background": color}),
        html.Span(text, className="opt-name"),
        html.Span("" if count is None else f"{count:,}", className="opt-count"),
    ], className="subtype-option")}


def sidebar() -> html.Aside:
    lo, hi = age_bounds()
    subtypes = checklist_options(CHECKLIST_FILTERS["f-subtype"])
    return html.Aside([
        html.Div([
            html.H2("Filters", className="sidebar-title"),
            html.P("Apply to every view on every page.", className="muted"),
        ]),
        html.Fieldset([
            html.Legend("Molecular subtype"),
            dcc.Checklist(id="f-subtype", options=[subtype_option(s, None) for s in subtypes],
                          value=subtypes, className="subtype-list"),
        ]),
        html.Fieldset([
            html.Legend("Receptor status"),
            *[html.Div([html.Span(name, className="receptor-name"),
                        _segmented(cid, [("All", "All"), ("Positive", "+"), ("Negative", "−")], "All")],
                       className="receptor-row")
              for cid, name in (("f-er", "ER"), ("f-pr", "PR"), ("f-her2", "HER2"))],
        ]),
        html.Fieldset([
            html.Legend("Age at diagnosis (years)"),
            dcc.RangeSlider(id="f-age", min=lo, max=hi, step=1, value=[lo, hi],
                            marks={v: str(v) for v in range(30, hi + 1, 20)},
                            tooltip={"placement": "bottom", "always_visible": False},
                            allowCross=False),
        ]),
        html.Fieldset([html.Legend(label("tumor_stage")), _chips("f-stage", "tumor_stage")]),
        html.Fieldset([html.Legend(label("neoplasm_histologic_grade")),
                       _chips("f-grade", "neoplasm_histologic_grade")]),
        html.Fieldset([html.Legend(label("vital_status")), _chips("f-vital", "vital_status")]),
        html.Fieldset([
            html.Legend("Treatment"),
            *[html.Div([html.Span(text), dcc.Dropdown(id=cid, options=["Any", "Yes", "No"], value="Any",
                                                       clearable=False, searchable=False,
                                                       className="mini-dropdown")],
                       className="treatment-row")
              for cid, text in (("f-chemo", "Chemotherapy"), ("f-hormone", "Hormone therapy"),
                                ("f-radio", "Radiotherapy"))],
        ]),
        html.Div([
            html.P(id="manual-filter-status", className="muted small"),
            html.Button("Clear all filters", id="clear-filters", className="btn btn-link"),
        ], className="sidebar-footer"),
    ], className="sidebar", **{"aria-label": "Filters"})


# --------------------------------------------------------------------------- #
# Selection bar (under the header, visible only when patients are selected)
# --------------------------------------------------------------------------- #
def selection_bar() -> html.Div:
    return html.Div([
        html.Span(id="selection-text", className="selection-text"),
        html.Div([
            html.Button("Keep only these", id="keep-selection", className="btn btn-small"),
            html.Button("Exclude these", id="exclude-selection", className="btn btn-small"),
            html.Button("Clear selection", id="clear-selection", className="btn btn-small btn-primary"),
        ], className="selection-actions"),
    ], id="selection-bar", className="selection-bar", role="status", style={"display": "none"})


# --------------------------------------------------------------------------- #
# Help panel
# --------------------------------------------------------------------------- #
HELP_SECTIONS = [
    ("1 · Filter", "The sidebar on the left removes patients from every chart on every page (subtype, "
                   "receptors, age, stage, grade, vital status, treatment). The header shows how many "
                   "patients remain; the number next to each subtype is how many patients it would add."),
    ("2 · Select", "Click a bar, a segment, a table cell or a box, drag a range on a histogram, or draw a "
                   "lasso on a scatter plot. The selected patients keep their colour in every chart, the "
                   "others turn grey or are shown as 'rest'. The blue bar under the header lets you clear "
                   "the selection, or keep only / exclude these patients (they then become a filter)."),
    ("3 · Read the details", "Hover any mark for exact values. The (i) button on each card explains how "
                             "to read the chart."),
    ("Overview", "Who are the patients and how do they survive? Key numbers, subtypes, survival curves "
                 "over the first 5 years, distribution of any measurement, composition of any category."),
    ("Relationships & genomics", "How are the variables linked? Click a cell of the matrix to open that "
                                 "pair below it; the pair view accepts clinical variables, genes and "
                                 "mutations. Which genes are mutated or expressed differently in a group?"),
    ("Clusters & projections", "Is there structure in the gene expression? Choose a projection and a "
                               "clustering algorithm, then press Run (t-SNE and UMAP take 10-30 s). The "
                               "clusters can then be used on the other pages ('by cluster')."),
    ("Good to know", "Molecular subtypes are the five PAM50 subtypes (LumA, LumB, Her2, Basal, Normal) "
                     "plus claudin-low, which METABRIC identifies with a separate classifier; NC = not "
                     "classified. Survival charts show, month by month, the share of patients still alive "
                     "among those whose status is known, over the first 5 years after diagnosis."),
]


def help_panel() -> html.Div:
    return html.Div([
        html.Div([
            html.Div([html.H2("How to use METABRIC Explorer"),
                      html.Button("Close", id="help-close", className="btn")], className="modal-head"),
            *[html.Div([html.H3(t), html.P(p)]) for t, p in HELP_SECTIONS],
        ], className="modal", role="dialog", **{"aria-modal": "true", "aria-label": "Help"}),
    ], id="help-panel", className="modal-backdrop", style={"display": "none"})
