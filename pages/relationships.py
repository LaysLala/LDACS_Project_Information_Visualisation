"""Page 2 - Relationships & genomics (built in Phase 3)."""

import dash
from dash import html

from src.components import card

dash.register_page(__name__, path="/relationships", name="Relationships & genomics",
                   title="Relationships & genomics · METABRIC Explorer", order=1)

layout = html.Div([
    card("Relationships & genomics", [
        html.P("This page is being built (Phase 3). It will show the association matrix "
               "(mutual information / Spearman), the adaptive pair view, mutation frequencies "
               "and the cluster heatmap. Filters and selections already apply.", className="muted"),
    ]),
], className="page")
