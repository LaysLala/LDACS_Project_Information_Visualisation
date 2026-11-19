"""
METABRIC Explorer: entry point.

    python app.py            then open http://127.0.0.1:8050

Layout: header (page tabs, patient counter, reset, help) + filter sidebar +
selection bar + the current page (pages/*.py, registered with Dash Pages).
"""

from __future__ import annotations

import sys

import dash
from dash import Dash, dcc, html

from src.data import DatasetMissingError, load_data
from src.state import EMPTY_MANUAL
from src.theme import register_template

try:
    load_data()  # read and clean the CSV once, at start-up (cached afterwards)
except DatasetMissingError as error:
    sys.exit(f"\n{error}\n")

register_template()

app = Dash(
    __name__,
    use_pages=True,
    title="METABRIC Explorer",
    suppress_callback_exceptions=True,
    external_stylesheets=["https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"],
)

from src import callbacks  # noqa: E402,F401  (registers the app-wide callbacks)
from src.components import header, help_panel, selection_bar, sidebar  # noqa: E402

app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="filtered-ids"),
    dcc.Store(id="selected", data=None),
    dcc.Store(id="manual-filter", data=EMPTY_MANUAL),
    header(),
    help_panel(),
    html.Div([
        sidebar(),
        html.Main([selection_bar(), dash.page_container], className="content"),
    ], className="body"),
], className="app")


if __name__ == "__main__":
    app.run(debug=False)
