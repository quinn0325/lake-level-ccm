"""Configuration for the whole pipeline.

Which lakes, which lags, which results file: all of it lives here so that no
script has to hard-code its own copy. The point was to stop the figures and the
forecasts silently drifting onto different branches of the results, which had
already happened once.
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

# ----------------------------------------------------------------- lags
# One signed scan over -12..+12 per edge, same for within-lake and between-lake.
# The observed series and every IAAFT surrogate go through the identical lag
# family; ccm_full_pipeline.CCM_LAGS is the only place the default is defined.
CAUSAL_LAGS = range(-12, 13)

# Forecast features must lag by at least one month: d=0 would need the driver's
# value at the forecast origin itself, which is not information you have.
# Causal identification and feature construction are two separate optimisation
# problems, each solved on its own lag domain.
FORECAST_LAGS = range(1, 13)

# --------------------------------------------------------------- inference
N_SURROGATES = 500
IAAFT_N_ITER = 100
FDR_ALPHA = 0.05

# ------------------------------------------------------------ missing data
MAX_FILLABLE_GAP_MONTHS = 6    # longer gap => drop the variable, don't interpolate
OUTLIER_Z_THRESH = 6
OUTLIER_LEVEL_Z_THRESH = 5

# -------------------------------------------------------------- results file
# Figures and forecasts take the filename from here, never hard-coded.
EMBED_PARAMS_FILE = RESULTS_DIR / "embed_params_corrected.json"

# Caveat: only the 00_data_generation/ scripts actually import this file. The
# rest still carry their own copies of the lake and variable lists. They agree,
# but editing here does not propagate everywhere -- check before assuming it
# does.
