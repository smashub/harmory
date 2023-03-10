"""
Default parameters for the creation of Harmory.
"""

SR = 1   # the symbolic sample rate

# NOVELTY CURVE
L_KERNEL = 8  # the relative size of the novelty kernel, in seconds
VAR_KERNEL = .5  # the variance of the Gaussian kernel for novelty computation

# PEAK DETECTION: MSAF
PDEC_MSAF_MEDIAN_LEN = 24
PDEC_MSAF_SIGMA = 2


# PATTERN SIMILARITY
RESAMPLING_SIZE = 30
N_SEARCHES = 10
DIST_THRESHOLD = 2.

# DTW PARAMETERISATION (validated empirically)
DTW_GLOBAL_CONSTRAINT = "sakoe_chiba"
DTW_SAKOE_RADIUS = 5

# ############################################################################ #
# PARAMETER-WISE GRID for search
# ############################################################################ #

