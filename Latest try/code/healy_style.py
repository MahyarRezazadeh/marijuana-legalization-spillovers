# ============================================================
# Minimal "Kieran Healy" style matplotlib theme: clean sans-serif,
# light horizontal gridlines only, no top/right spines, muted palette,
# direct labeling favored over boxed legends. Import and call
# apply_healy_style() before creating figures.
# ============================================================
import matplotlib.pyplot as plt

PALETTE = {
    'red': '#B33951',
    'blue': '#2A6F97',
    'gray': '#7C7C7C',
    'dark': '#2B2B2B',
    'gold': '#C98F2B',
}


def apply_healy_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.edgecolor': '#4D4D4D',
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'axes.axisbelow': True,
        'grid.color': '#E3E3E3',
        'grid.linewidth': 0.7,
        'xtick.color': '#4D4D4D',
        'ytick.color': '#4D4D4D',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.frameon': False,
        'legend.fontsize': 9.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 200,
    })
