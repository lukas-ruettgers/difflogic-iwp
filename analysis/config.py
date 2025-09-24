SAVE_PDF = True
SAVE_PNG = False
SAVE_PGF = False

import shutil
import sys
# Check for LaTeX installation
LATEX_BINARIES = ["latex", "pdflatex", "xelatex"]
if not all(shutil.which(bin) for bin in LATEX_BINARIES):
    raise RuntimeError(
        "\n[ERROR] No LaTeX engine found.\n"
        "To use LaTeX rendering in matplotlib, install the following:\n"
        "  - latex\n"
        "  - pdflatex\n"
        "  - xelatex\n"
        )

import zarr
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter, LogLocator, NullFormatter
import numpy as np
import seaborn as sns
import os, sys
import itertools

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import library
import importlib
importlib.reload(library)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from execution_setup.directories import DIR_RESULTS, DIR_ANALYSIS

# Set publication-quality style with seaborn
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
    
LATEX_WIDTH_INCH = 4.8
HEIGHT_TO_WIDTH_FACTOR = 0.7
DEFAULT_FIG_SIZE = (LATEX_WIDTH_INCH, HEIGHT_TO_WIDTH_FACTOR*LATEX_WIDTH_INCH)

# Use LaTeX for all text rendering
mpl.rcParams.update({
    "text.usetex": True,              # Enable thes only if LaTeX and pdflatex is installed
    "pgf.texsystem": "pdflatex",
    "text.latex.preamble": r"\usepackage{amsmath}",  # Optional: for math
    "font.family": "serif",             # Match LaTeX serif fonts
    "font.serif": ["Times New Roman", "Times"],  # Use Times to match LaTeX default
    "axes.labelsize": 11,               # Match your LaTeX font size
    "font.size": 11,                    # Global font size
    "legend.fontsize": 14,              # Slightly smaller legend
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

def load_zarr(fname, results_dir=DIR_RESULTS):
    """Load weights for a specific layer from saved zarr arrays"""
    zarr_path = Path(results_dir) / fname
    
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr array not found: {zarr_path}")
    
    a = zarr.open(zarr_path, mode='r')
    return a