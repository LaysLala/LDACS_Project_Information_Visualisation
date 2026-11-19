"""Page 3 - Clusters & projections (built in Phase 4)."""

import dash
from dash import html

from src.components import card

dash.register_page(__name__, path="/clusters", name="Clusters & projections",
                   title="Clusters & projections · METABRIC Explorer", order=2)

layout = html.Div([
    card("Clusters & projections", [
        html.P("This page is being built (Phase 4). It will show PCA / t-SNE / UMAP projections, "
               "K-means, hierarchical and DBSCAN clustering with their quality measures, and survival "
               "per cluster. Filters and selections already apply.", className="muted"),
    ]),
], className="page")
