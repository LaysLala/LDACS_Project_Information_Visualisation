"""
App-wide callbacks: filters -> filtered ids, selection bar, header counter,
Keep only / Exclude / Reset, help panel and active page tab.

Page-specific callbacks live in their page module (pages/*.py).
"""

from __future__ import annotations

from dash import Input, Output, State, callback, ctx, no_update

from src.components import PAGES, subtype_option
from src.data import load_data
from src.state import (CHECKLIST_FILTERS, EMPTY_MANUAL, RECEPTOR_FILTERS, TREATMENT_FILTERS,
                       apply_filters, checklist_options, default_filters, visible_selection)

FILTER_IDS = list(CHECKLIST_FILTERS) + list(RECEPTOR_FILTERS) + list(TREATMENT_FILTERS) + ["f-age"]
FILTER_INPUTS = [Input(cid, "value") for cid in FILTER_IDS]


@callback(Output("filtered-ids", "data"), *FILTER_INPUTS, Input("manual-filter", "data"))
def update_filtered(*args):
    *values, manual = args
    filters = dict(zip(FILTER_IDS, values))
    return [int(i) for i in apply_filters(filters, manual)]


@callback(Output("f-subtype", "options"), *[Input(cid, "value") for cid in FILTER_IDS if cid != "f-subtype"],
          Input("manual-filter", "data"))
def subtype_counts(*args):
    """Scented widget: next to each subtype, how many patients it has under the other filters."""
    *values, manual = args
    filters = dict(zip([c for c in FILTER_IDS if c != "f-subtype"], values))
    ids = apply_filters(filters, manual, skip="f-subtype")
    counts = load_data().clinical.loc[ids, "pam50_subtype"].astype(str).value_counts()
    return [subtype_option(s, int(counts.get(s, 0))) for s in checklist_options("pam50_subtype")]


@callback(Output("patient-counter", "children"), Input("filtered-ids", "data"), Input("selected", "data"))
def counter(filtered, selected):
    total = load_data().n_patients
    n = len(filtered or [])
    text = [f"{n:,} of {total:,} patients"]
    n_sel = len(visible_selection(selected, filtered or []))
    if n_sel:
        text.append(f" · {n_sel:,} selected")
    return "".join(text)


@callback(Output("selection-bar", "style"), Output("selection-text", "children"),
          Input("selected", "data"), Input("filtered-ids", "data"))
def selection_bar(selected, filtered):
    ids = visible_selection(selected, filtered or [])
    if not ids:
        return {"display": "none"}, ""
    hidden = len(selected["ids"]) - len(ids)
    extra = f" ({hidden:,} more hidden by the filters)" if hidden else ""
    return {}, [f"{len(ids):,} patients selected", f" · {selected.get('source', '')}{extra}. ",
                "Selected patients keep their colour in every chart; the others are greyed or shown as 'rest'."]


@callback(Output("selected", "data", allow_duplicate=True), Input("clear-selection", "n_clicks"),
          prevent_initial_call=True)
def clear_selection(_):
    return None


@callback(Output("manual-filter", "data", allow_duplicate=True), Output("selected", "data", allow_duplicate=True),
          Input("keep-selection", "n_clicks"), Input("exclude-selection", "n_clicks"),
          State("selected", "data"), State("filtered-ids", "data"), State("manual-filter", "data"),
          prevent_initial_call=True)
def selection_to_filter(_keep, _exclude, selected, filtered, manual):
    ids = visible_selection(selected, filtered or [])
    if not ids:
        return no_update, no_update
    manual = dict(manual or EMPTY_MANUAL)
    if ctx.triggered_id == "keep-selection":
        manual["keep"] = ids
    else:
        manual["exclude"] = sorted(set(manual.get("exclude") or []) | set(ids))
    return manual, None


@callback(Output("manual-filter-status", "children"), Input("manual-filter", "data"))
def manual_status(manual):
    manual = manual or EMPTY_MANUAL
    parts = []
    if manual.get("keep") is not None:
        parts.append(f"Keeping only {len(manual['keep']):,} previously selected patients.")
    if manual.get("exclude"):
        parts.append(f"{len(manual['exclude']):,} patients excluded.")
    return " ".join(parts)


@callback([Output(cid, "value", allow_duplicate=True) for cid in FILTER_IDS],
          Output("manual-filter", "data", allow_duplicate=True),
          Output("selected", "data", allow_duplicate=True),
          Input("reset-all", "n_clicks"), Input("clear-filters", "n_clicks"),
          prevent_initial_call=True)
def reset(_all, _filters):
    defaults = default_filters()
    values = [defaults[cid] for cid in FILTER_IDS]
    selection = None if ctx.triggered_id == "reset-all" else no_update
    return *values, EMPTY_MANUAL, selection


@callback(Output("help-panel", "style"), Input("help-open", "n_clicks"), Input("help-close", "n_clicks"),
          prevent_initial_call=True)
def toggle_help(_open, _close):
    return {} if ctx.triggered_id == "help-open" else {"display": "none"}


@callback([Output(f"nav-{i}", "className") for i in range(len(PAGES))], Input("url", "pathname"))
def active_tab(pathname):
    return ["nav-link active" if pathname == path else "nav-link" for path, _ in PAGES]
