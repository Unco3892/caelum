"""Shared plotting defaults for the original CAELUM analyses.

Usage:
    from caelum.plotting import apply_paper_style, COLORS, METHOD_COLORS, SIGNAL_COLORS
    apply_paper_style()

Theme:
    Serif body + STIX math (matches LaTeX output), 300 dpi savefig, hidden top/right
    spines, thin axes, colorblind-safe Tol-bright palette.
"""

from __future__ import annotations

import matplotlib as mpl


# Colorblind-friendly palette (Paul Tol bright)
COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377", "#66CCEE", "#BBBBBB"]

# Method -> color (consistent across all CAELUM figures)
METHOD_COLORS = {
    "CAELUM":            "#4477AA",
    "Vanilla":           "#BBBBBB",
    "Vanilla CP":        "#BBBBBB",
    "Equal":             "#66CCEE",
    "EqualWeight":       "#66CCEE",
    "Equal-weight":      "#66CCEE",
    "EpiOnly":           "#AA3377",
    "EpistemicOnly":     "#AA3377",
    "Epistemic-only":    "#AA3377",
    "ConU":              "#EE6677",
    "SE-CP":             "#CCBB44",
    "SemanticEntropyCP": "#CCBB44",
    "TrueAPS":           "#228833",
}

# Uncertainty-signal -> color
SIGNAL_COLORS = {
    "U_total": "#BBBBBB",
    "U_epi":   "#4477AA",
    "U_ale":   "#CCBB44",
    "R(x)":    "#228833",
}


def apply_paper_style() -> None:
    """Apply the publication-quality rcParams. Call once at module import time."""
    mpl.rcParams.update(
        {
            "font.family":        "serif",
            "mathtext.fontset":   "stix",
            "font.size":          10,
            "axes.labelsize":     11,
            "axes.titlesize":     12,
            "legend.fontsize":    9,
            "xtick.labelsize":    9,
            "ytick.labelsize":    9,
            "savefig.dpi":        300,
            "figure.dpi":         150,
            "axes.linewidth":     0.8,
            "axes.spines.top":    False,
            "axes.spines.right":  False,
            "grid.linewidth":     0.5,
            "lines.linewidth":    1.8,
            "lines.markersize":   6,
            "savefig.bbox":       "tight",
            "savefig.pad_inches": 0.05,
        }
    )
