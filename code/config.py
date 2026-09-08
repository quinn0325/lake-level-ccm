"""Shared configuration for the data-generation stage.

Caveat: only the 00_data_generation/ scripts import this file. The analysis and
figure scripts carry their own copies of the lake list and the lag ranges. They
agree, and verify.py would catch it if they stopped agreeing, but this is not
the single source of truth it was meant to be. Values that nothing outside this
file read have been removed rather than left here looking authoritative.
"""

from pathlib import Path

# ------------------------------------------------------------------ paths
CODE_DIR = Path(__file__).resolve().parent
PKG_DIR = CODE_DIR.parent
RESULTS_DIR = PKG_DIR / "results"
PKL_DIR = PKG_DIR / "lake_pkls"

# ------------------------------------------------------------- study system
LAKES = [
    # Okanagan system (British Columbia)
    "Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake", "Vaseux_Lake",
    # Nelson-Winnipeg system (Manitoba / Ontario)
    "Rainy_Lake", "Lake_of_the_Woods", "Playgreen_Lake",
    "Kiskitto_Lake", "Sipiwesk_Lake", "Split_Lake",
]

VARIABLES = ["WL", "T", "P", "R", "SWE", "Evap", "RegFlow"]

# The 7 pairs joined by a direct waterway, used as the connectivity contrast.
WATERWAY_CONNECTED_PAIRS = [
    ("Kalamalka_Lake", "Okanagan_Lake"),
    ("Okanagan_Lake", "Skaha_Lake"),
    ("Skaha_Lake", "Vaseux_Lake"),
    ("Rainy_Lake", "Lake_of_the_Woods"),
    ("Playgreen_Lake", "Sipiwesk_Lake"),
    ("Kiskitto_Lake", "Sipiwesk_Lake"),
    ("Sipiwesk_Lake", "Split_Lake"),
]

# --------------------------------------------------------------- train/test
FORECAST_HORIZON = 37          # roughly 9:1. CCM also sees only the training part.
ROLLING_HORIZONS = [1, 3, 6, 12]

# ------------------------------------------------------------ embedding
# tau fixed at 1, following Javier et al. (2022, Physica A 604, 127893):
# "We fixed tau = 1 because this gives the highest resolution for the embedding".
# Fixing it also removes one route for test-period information to leak in through
# parameter selection, and shrinks the embedding span (E-1)*tau from up to 36
# months to at most 9, which matters a lot on the series with gaps.
# E is still chosen by univariate simplex self-prediction. Deliberately not by
# cross-map skill: choosing the parameter with the quantity under test is
# circular.
EMBED_TAU = 1
EMBED_E_CANDIDATES = range(2, 11)   # E ∈ [2, 10]

# --------------------------------------------------------------- inference
N_SURROGATES = 500
FDR_ALPHA = 0.05

# ------------------------------------------------------------ missing data
MAX_FILLABLE_GAP_MONTHS = 6    # longer gap => drop the variable, don't interpolate
OUTLIER_Z_THRESH = 6
OUTLIER_LEVEL_Z_THRESH = 5

# Caveat: only the 00_data_generation/ scripts actually import this file. The
# rest still carry their own copies of the lake and variable lists. They agree,
# but editing here does not propagate everywhere -- check before assuming it
# does.
