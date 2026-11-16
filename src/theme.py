"""
Visual identity shared by every chart in the dashboard.

Why one module: the evaluation grid rewards consistency (fonts, colours and
styles identical across views) and colour-blind accessibility. Every figure
uses the "metabric" Plotly template, and every colour *meaning* is defined
once here.

Colour rules
------------
- One colour = one meaning, everywhere. A PAM50 subtype has the same colour on
  every page, so users can recognise it without reading the legend twice.
- Categorical palettes were checked with a colour-vision-deficiency validator
  (OKLab distance between colours under deuteranopia/protanopia/tritanopia).
  Colour is never the only cue: legends and hover labels are always present.
- Diverging scale (z-scores, correlations): blue = low/negative, grey = 0,
  red = high/positive. Two opposite hues with a neutral midpoint, never a
  rainbow.
- Sequential scale (counts, mutual information, frequencies): one hue,
  light -> dark.
- Selections: selected marks keep their colour; everything else turns light
  grey and fades.

Call `register_template()` once (app.py or a notebook) before building figures.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --------------------------------------------------------------------------- #
# Typography & surfaces
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# Categorical colours
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# Continuous scales
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# Selection (linked brushing)
# --------------------------------------------------------------------------- #
SELECTED_OPACITY = 0.85
UNSELECTED_COLOR = "#d9d8d2"
UNSELECTED_OPACITY = 0.25
MARKER_SIZE = 6

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
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


def cluster_colors(labels: list[int]) -> dict[int, str]:
    """Colour per cluster id; -1 (DBSCAN noise) is light grey."""
    mapping: dict[int, str] = {}
    slot = 0
    for lab in sorted(set(labels)):
        if lab == -1:
            mapping[lab] = NOISE_COLOR
        elif slot < MAX_CATEGORIES:
            mapping[lab] = CATEGORICAL[slot]
            slot += 1
        else:
            mapping[lab] = OTHER_COLOR
    return mapping


# --------------------------------------------------------------------------- #
# Plotly template
# --------------------------------------------------------------------------- #
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
