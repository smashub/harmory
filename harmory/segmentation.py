"""
Harmonic Structure Analysis via cofnitive and musicologically-plausible self
similarity matrices (SSM) defined from the Tonal Pitch Space (TPS).

Notes: XXX this will be moved to a separate [library currently in preparation].
"""
import logging

import numpy as np
from scipy.ndimage import filters

from tonalspace import TpsComparator

logger = logging.getLogger("segmentation")


def compute_sm_dot(X, Y):
    """
    Computes similarty matrix from feature sequences using dot (inner) product.

    Parameters
    ----------
    X : np.ndarray
        The first sequence to consider for comparison.
    Y : np.ndarray
        The second sequence to consider for comparison.

    Returns
    -------
    S : np.array
        The SM computed as dot product of X and Y.

    """
    S = np.dot(np.transpose(X), Y)
    return S


def create_chord_ssm(chords: list[str], keys: list[str],
                     normalisation=True, as_distance=False, symmetric=True):
    """
    Creation of the self-similarity matrix from chord-key annotations. This
    implementation performs pairwise chord comparisons using the Tonal Pitch
    Step Distance between (chord, key) pairs. Complexity is thus quadratic in
    the sequence length, and computation is already optimised to avoid repeated
    comparisons for quantised durations. This is why a temporal SSM can then be
    obtained using `expand_ssm()` before any further operation is applied.

    Parameters
    ----------
    chords : list of str
        A list of chords in Harte notation, aligned with the given keys.
    keys : list of str
        A list of local keys that are assumed to be aligned with `chords`.
    normalisation : bool
        Whether the matrix should be normalised w.r.t. the TPSD range.
    as_distance : bool
        Wheteher to use distances or similarity values for the scores.
    symmetric : bool
        Whether forcing the TPSD to return symmetric values; otherwise, for
        some chord pairs s(i, j) != s(j, i) due to the original TPSD.

    Returns
    -------
    ssm : np.array
        A 2-dimensional array containing the self-similarity matrix. Note, that
        the returned SSM needs to be expanded to take durations into account --
        unless this has already been done (`chords` and `keys` contain quantised
        steps).
    tpsd_cache : dict
        A map containing the raw TPS distances among all given chord-key pairs. 

    See also
    --------
    expand_ssm()

    Notes
    -----
    - Computation can be speeded up if saving a dictionary/mapping of chord-key
        to chord distances. This can be even kept in memory and updated for the
        SSM computation of other tracks.
    - Another trick that can save computation is to compress all consecutive
        chords with the same label, as this produces trivial repetitions. Not
        needed if the former step is implemented.
    - Indexes of SSM can be obtained from cartesian product of chord-key
        occurrences after quantisation.

    """
    tps_comparator = TpsComparator()
    chord_ssm = np.zeros((len(chords), len(chords)))
    np.fill_diagonal(chord_ssm, 0)  # minimum distance among same occurrences

    for i in range(len(chords) - 1):
        chord_i, key_i = chords[i], keys[i]
        for j in range(i+1, len(chords)): # look ahead
            chord_j, key_j = chords[j], keys[j]
            # logger.debug(f"TPSD distance d({i},{j})")
            tpsd_ij = tps_comparator.tpsd_lookup(
                chord_a=chord_i, key_a=key_i,
                chord_b=chord_j, key_b=key_j)
            tpsd_ji = tps_comparator.tpsd_lookup(
                chord_a=chord_j, key_a=key_j,
                chord_b=chord_i, key_b=key_i
            )
            # Update the matrix, either symmetrically or asymmetrically
            chord_ssm[i, j], chord_ssm[j, i] = (tpsd_ij + tpsd_ji) / 2 \
                if symmetric else tpsd_ij, tpsd_ji

    # Normalisation w.r.t. TPSD range [0, 13] and to sim, if required
    if normalisation:
        chord_ssm = chord_ssm / 13
    # Similarity values are converted as distance values, if required
    if as_distance is False:
        chord_ssm = 1 - chord_ssm

    return chord_ssm, tps_comparator._tpsd_cache


def expand_ssm(ssm: np.array, times: list):
    """
    Expand a symbolic self-similarity matrix.

    Parameters
    ----------
    ssm : np.array
        A 2-dimensional array encoding the original self-similarity matrix.
    times : list
        List of quantised observation onsets, including the end frame. Each
        element corresponds to the time of a row/column in the SSM.

    Returns
    -------
    ssm_full : np.array
        The self-similarity matrix expanded according to the quantisation. 
 
    """
    if ssm.shape[0] != len(times) - 1:
        raise ValueError("SSM-times mismatch: times should describe entries!" \
                        f" Expected {ssm.shape[0]}; Found {len(times)}")
    # SSM temporal expansion following quantisation
    qtimes = times[:-1] + [times[-1] + 1]  # include last
    ssm_full = np.zeros((qtimes[-1], qtimes[-1]))

    windows = list(zip(qtimes[:-1], qtimes[1:]))

    for i, window in enumerate(windows):
        anchor = ssm[i]  # select the anchor
        assert len(windows) == len(anchor)
        # Broadcasting within the anchor before
        for infill, sim in zip(windows, anchor):
            ssm_full[window[0]][infill[0]:infill[1]] = sim
        # Fixing similarity to 1 for the repeated entries
        ssm_full[window[0]][window[0]:window[1]] = 1
        # Broadcasting the anchor vector to the repeated entries
        ssm_full[window[0]:window[1]] = ssm_full[window[0]] 

    return ssm_full


# ############################################################################ #
# Kernel generators and novelty computation
# ############################################################################ #

def compute_kernel_checkerboard_box(L):
    """
    Generate a box-like checkerboard kernel where values in the upper right and
    the lower left quadrants are all 1s, and the others are all set to 0. Mostly
    used for demonstrational purposes.

    Parameters
    ----------
    L : int
        Parameterises the size of the box-like kernel (a square), which has 
        actual size of (2*L+1) x (2*L+1).

    Returns
    -------
    kernel : np.ndarray
        A box-like kernel matrix that is parameterised by L.

    """
    axis = np.arange(-L, L+1)
    kernel = np.outer(np.sign(axis), np.sign(axis))
    return kernel


def compute_kernel_checkerboard_gaussian(L, var=1.0, normalize=True):
    """
    Generate a Guassian-like checkerboard kernel
    
    Parameters
    ----------
    L : int
        Parameter specifying the kernel size M=2*L+1.
    var : float
        Variance parameter determing the tapering (epsilon) of the kernel.
    normalize : bool
        Whether to normalize kernel values.

    Returns
    -------
    kernel : np.ndarray
        Kernel matrix of size M x M, where M = 2*L+1.

    See also
    --------
    https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

    """
    taper = np.sqrt(1/2) / (L * var)
    axis = np.arange(-L, L+1)
    gaussian1D = np.exp(-taper**2 * (axis**2))
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    kernel_box = np.outer(np.sign(axis), np.sign(axis))
    kernel = kernel_box * gaussian2D
    if normalize:  # normalising kernel values based on range
        kernel = kernel / np.sum(np.abs(kernel))
    return kernel


def leaky_kernel(L, var=.5, lprob=.2, normalize=True):
    """
    Return a leaky kernel where rows/cols are zeroed-out depending on lprob.
    This kernel is experimental and should not be used for segmentation.

    Parameters
    ----------
    L : int
        Parameter specifying the kernel size M=2*L+1.
    var : float
        Variance parameter determing the tapering (epsilon) of the kernel.
    lprob : float
        The probability of zeroing a row/column in the Gaussian kernel.
    
    Returns
    -------
    kernel : np.ndarray
        Kernel matrix of size M x M, where M = 2*L+1.

    """
    K = compute_kernel_checkerboard_gaussian(L=L, var=.5, normalize=True)
    off_no = round(K.shape[0] * lprob)  # the number of rows/cols to zero out
    off_rows = np.random.randint(0, K.shape[0], size=off_no)
    K[:, off_rows] = K[off_rows, :] = 0
    return K


def compute_novelty_sm(S, kernel=None, L=10, var=0.5, exclude=False):
    """
    Compute a novelty function from the SM, by correlating the given kernel
    alongside the main diagonal of the matrix.

    Parameters
    ----------
    S : np.ndarray
        The similarity matrix that will be processed for novelty.
    kernel : np.ndarray
        A kernel to convolute on the SM (if `None`, it will be created).
    L : int
        If no kernel is specified, and a new one is created, this paramter
        controls the kernel size (defined by M=2*L+1).
    var : float:
        Variance parameter determing the tapering of the kermnel (epsilon).
    exclude : bool
        Whether to set the first L and last L values of the novelty function to
        zero, in order to account for the kernel size at the extremes. This
        simply corresponds to padding the similarity matrix.

    Returns
    -------
    nov : np.ndarray
        The novelty function computed on `S` using `kernel`.

    """
    if kernel is None:
        kernel = compute_kernel_checkerboard_gaussian(L=L, var=var)
    N = S.shape[0]
    M = 2*L + 1
    nov = np.zeros(N)
    # XXX np.pad does not work with numba/jit
    S_padded = np.pad(S, L, mode='constant')

    for n in range(N):
        # XXX np.pad does not work with numba/jit
        nov[n] = np.sum(S_padded[n:n+M, n:n+M] * kernel)
    if exclude:
        right = np.min([L, N])
        left = np.max([0, N-L])
        nov[0:right] = 0
        nov[left:N] = 0

    return nov


# ############################################################################ #
# Utilities for peak detection, see also:
# - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html 
# - https://librosa.org/doc/latest/generated/librosa.util.peak_pick.html
# ############################################################################ #

def peak_picking_simple(x, threshold=None):
    """
    Peak picking looking for positions with increase followed by descrease. This
    corresponds to finding the local maxima in a discrete function. A global
    threshold is used to exclude all picks whose novelty falls below a minimum.

    Parameters
    ----------
    x : np.ndarray
        The input novelty curve from which peak locations are estimated.
    threshold : float
        Lower threshold to retain a peak after candidate selection.

    Returns
    -------
    peaks : np.ndarray
        An array containing the location of all the detected peaks.

    """
    peaks = []
    if threshold is None:
        threshold = np.min(x) - 1
    for i in range(1, x.shape[0] - 1):
        if x[i - 1] < x[i] and x[i] > x[i + 1]:
            if x[i] >= threshold:
                peaks.append(i)
    peaks = np.array(peaks)
    return peaks


def peak_picking_boeck(x, threshold=0.5, fps=100, include_scores=False, combine=False,
                       pre_avg=12, post_avg=6, pre_max=6, post_max=6):
    """
    Peak detection based on Boeck's algorithm for beat onset estimation.

    This function implements the peak-picking method described in:
      "Evaluating the Online Capabilities of Onset Detection Methods" by 
      Sebastian Boeck, Florian Krebs and Markus Schedl, in Proceedings of the
      13th International Society for Music Information Retrieval Conference

    Parameters
    ----------
    x : np.ndarray
        The input novelty curve from which peak locations are estimated.
    threshold : float
        A threshold for peak-picking.
    fps : scalar
        Frame rate of onset activation function in Hz.
    include_scores : bool
        Include activation for each returned peak.
    combine : bool
        Only report 1 onset for N seconds.
    pre_avg : float
        Use N past seconds for moving average.
    post_avg : float
        Use N future seconds for moving average.
    pre_max : float
        Use N past seconds for moving maximum.
    post_max : float
        Use N future seconds for moving maximum.

    Returns
    -------
    peaks : np.ndarray
        A nunpy array with the detected peak positions.

    """
    activations = x.ravel()
    # Detections are activations equal to the moving maximum
    max_length = int((pre_max + post_max) * fps) + 1

    if max_length > 1:
        max_origin = int((pre_max - post_max) * fps / 2)
        mov_max = filters.maximum_filter1d(
            activations, max_length, mode='constant', origin=max_origin)
        detections = activations * (activations == mov_max)
    else:
        detections = activations

    # Detections must be greater than or equal to the moving average + threshold
    avg_length = int((pre_avg + post_avg) * fps) + 1
    if avg_length > 1:
        avg_origin = int((pre_avg - post_avg) * fps / 2)
        mov_avg = filters.uniform_filter1d(
            activations, avg_length, mode='constant', origin=avg_origin)
        detections = detections * (detections >= mov_avg + threshold)
    else:
        # If there is no moving average, treat the threshold as a global one
        detections = detections * (detections >= threshold)

    # Convert detected onsets to a list of timestamps
    if combine:
        stamps = []
        last_onset = 0
        for i in np.nonzero(detections)[0]:
            # Only report an onset if the last N frames none was reported
            if i > last_onset + combine:
                stamps.append(i)
                # Save last reported onset
                last_onset = i
        stamps = np.array(stamps)
    else:
        stamps = np.where(detections)[0]

    # Include corresponding activations per peak if needed
    if include_scores:
        scores = activations[stamps]
        if avg_length > 1:
            scores -= mov_avg[stamps]
        return stamps / float(fps), scores
    else:
        return stamps / float(fps)


def peak_picking_roeder(x, direction=None, abs_thresh=None, rel_thresh=None,
                        descent_thresh=None, tmin=None, tmax=None):
    """
    Computes the positive peaks of a novelty curve based on the `peaks.m`
    implementation of the Matlab Roeder_Peak_Picking from the Sync Toolbox.

    Parameters
    ----------
    x : np.ndarray
        The input novelty curve from which peak locations are estimated.
    direction : int
        Direction for peak detection, either +1 for forward peak searching, or
        -1 for backward peak searching.
    abs_thresh : float
        The absolute threshold signal, i.e. only peaks that meet this condition
        `x(i)>=abs_thresh(i)` will be considered. This parameter must have the 
        same number of samples as `x`. A sensible choice for this parameter
        would be a global or local average or median of the signal `x`. If
        omitted, half the median of `x` will be used.
    rel_thresh : float
        Relative threshold signal. Only peak positions i with an uninterrupted
        positive ascent before position `i` of at least `rel_thresh(i)` and a
        possibly interrupted (see parameter descent_thresh) descent of at least
        `rel_thresh(i)` will be reported. `rel_thresh` must have the same number
        of samples as x. A sensible choice would be some measure related to the
        global or local variance of the signal x. If omitted, half the standard
        deviation of W will be used.
    descent_thresh : float
        The descent threshold to consider. During peak candidate verfication, if
        a slope change from negative to positive slope occurs at sample i BEFORE
        the descent has exceeded `rel_thresh(i)`, and if `descent_thresh(i)` has
        not been exceeded yet, the current peak candidate will be dropped. This
        situation corresponds to a secondary peak occuring shortly after the
        current candidate peak (which might lead to a higher peak value). The
        value `descent_thresh(i)` must not be larger than `rel_thresh(i)`. Also,
        `descent_thresh` must have the same number of samples as `x`. A sensible
        choice would be some measure related to the global or local variance of
        the signal `x`. If omitted, `0.5*rel_thresh` will be used.
    tmin : int
        Index of the start sample. Peak search will thus begin at `x(tmin)`.
    tmax : int
        Index of the end sample. Peak search will end at `x(tmax)`.

    Returns
    -------
    peaks : np.ndarray
        A nunpy array with the detected peak positions.

    """
    # set default values
    if direction is None:
        direction = -1
    if abs_thresh is None:
        abs_thresh = np.tile(0.5*np.median(x), len(x))
    if rel_thresh is None:
        rel_thresh = 0.5*np.tile(np.sqrt(np.var(x)), len(x))
    if descent_thresh is None:
        descent_thresh = 0.5*rel_thresh
    if tmin is None:
        tmin = 1
    if tmax is None:
        tmax = len(x)

    dyold = 0
    dy = 0
    rise = 0  # current amount of ascent during a rising portion of the signal x
    riseold = 0  # accumulated amount of ascent from the last rising portion of x
    descent = 0  # current amount of descent (<0) during a falling portion of x
    searching_peak = True
    candidate = 1
    P = []

    if direction == 1:
        my_range = np.arange(tmin, tmax)
    elif direction == -1:
        my_range = np.arange(tmin, tmax)
        my_range = my_range[::-1]

    # run through x
    for cur_idx in my_range:
        # get local gradient
        dy = x[cur_idx+direction] - x[cur_idx]

        if dy >= 0:
            rise = rise + dy
        else:
            descent = descent + dy

        if dyold >= 0:
            if dy < 0:  # slope change positive->negative
                if rise >= rel_thresh[cur_idx] and searching_peak is True:
                    candidate = cur_idx
                    searching_peak = False
                riseold = rise
                rise = 0
        else:  # dyold < 0
            if dy < 0:  # in descent
                if descent <= -rel_thresh[candidate] and searching_peak is False:
                    if x[candidate] >= abs_thresh[candidate]:
                        P.append(candidate)  # verified candidate as True peak
                    searching_peak = True
            else:  # dy >= 0 slope change negative->positive
                if searching_peak is False:  # currently verifying a peak
                    if x[candidate] - x[cur_idx] <= descent_thresh[cur_idx]:
                        rise = riseold + descent  # skip intermediary peak
                    if descent <= -rel_thresh[candidate]:
                        if x[candidate] >= abs_thresh[candidate]:
                            P.append(candidate)  # verified candidate as True peak
                    searching_peak = True
                descent = 0
        dyold = dy
    peaks = np.array(P)
    return peaks


def peak_picking_msaf(x, median_len=16, offset_rel=0.05, sigma=4.0):
    """
    Peak picking strategy using adaptive threshold (as used in MSFAF).

    A smoothing filter is applied to the novelty function to reduce the effect
    of noise-like fluctuations. Furthermore, instead of considering a global
    threshold for discarding small, noise-like peaks, an adaptive thresholding
    is used to select a peak only when its value exceeds a local average of the
    novelty function in that neighbourhood.

    Parameters
    ----------
    x : np.ndarray
        The input novelty curve from which peak locations are estimated.
    median_len : int
        Length of the media filter used for adaptive thresholding.
    offset_rel : float
        Additional offset used for adaptive thresholding.
    sigma : float
        Variance of the Gaussian kernel used for smoothing the novelty function.

    Returns
    -------
    peaks : np.ndarray
        A nunpy array with the detected peak positions.
    threshold_local : np.ndarray
        The local threshold computed from the novelty curve.
    x : np.ndarray
        The novelty curve obtained after filtration.

    See also
    --------
    https://github.com/urinieto/msaf

    """
    offset = x.mean() * offset_rel
    x = filters.gaussian_filter1d(x, sigma=sigma)
    threshold_local = filters.median_filter(x, size=median_len) + offset
    peaks = []
    for i in range(1, x.shape[0] - 1):
        if x[i - 1] < x[i] and x[i] > x[i + 1]:
            if x[i] > threshold_local[i]:
                peaks.append(i)
    peaks = np.array(peaks)
    return peaks, x, threshold_local


# TODO Make this a factory static class
PDETECTION_MAP = {
    "simple"  : peak_picking_simple,
    "boek"    : peak_picking_boeck,
    "roeder"  : peak_picking_roeder,
    "msaf"    : peak_picking_msaf,
    "scipy"   : None,  # TODO
    "librosa" : None,  # TODO
}

