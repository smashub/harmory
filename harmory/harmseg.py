"""
Classes and methods for harmonic segmentation.

"""
import os
import copy
import pickle
import logging

import jams
import stumpy
import numpy as np

import segmentation as seg
from tonalspace import TpsOffsetTimeSeries, TpsProfileTimeSeries
from data import create_chord_sequence, postprocess_chords, postprocess_keys

logger = logging.getLogger("harmory.harmseg")


class HarmonicPrint:
    """
    The harmonic print of a progression holds relevant representations.

    Notes
    - This may contain both the TPS time series types.
    """
    def __init__(self, jams_path: str, sr: int,
                 tpst_type: str = "offset",
                 chord_namespace: str = "chord",
                 key_namespace: str = "key_mode") -> None:
        """
        Harmonic sequence encoder from a JAMS file and symbolic sample rate.
        """
        self.sr = sr  # the symbolic sample rate
        self.id = os.path.splitext(os.path.basename(jams_path))[0]  # jaah_10
        jams_object = jams.load(jams_path, validate=False)
        # Sanity check of the requested TPS time series tpye against supported
        if tpst_type not in ["offset", "profile"]:  # supported TPS time series
            raise ValueError(f"Not a supported time series type: {tpst_type}")
        self._tpst_type = tpst_type
        # Sanity checks on the JAMS file: chord and key annotations expected
        if len(jams_object.annotations.search(namespace=chord_namespace)) == 0:
            raise ValueError("No chord annotation in JAMS file!")            
        self.metadata = jams_object.file_metadata  # just keep meta?
        # Extracting, processing, and aligning chords, keys, and times
        chords, keys, self._times = \
             create_chord_sequence(jams_object, 1 / sr,
                                   shift=True,
                                   chord_namespace=chord_namespace,
                                   key_namespace=key_namespace)
        # XXX The following should be done at JAMS-loading stage
        self._chords = postprocess_chords(chords=chords)
        self._keys = postprocess_keys(keys=keys)
        # Class object that will be created on demand
        self._chord_ssm, self._tpsd_cache = None, None
        self._tps_timeseries = None

    def run(self):
        """
        Populates the data structures of this HarmonicPrint. This is kept
        separate as it allows for the inspection of intermediate variables.
        """
        self._chord_ssm, self._tpsd_cache = self._compute_chord_ssm()
        self._tps_timeseries = self._compute_tps_timeseries()

    @property
    def chord_ssm(self):
        return self._chord_ssm  # XXX should return a copy
    
    @property
    def tps_timeseries(self):
        return self._tps_timeseries  # XXX should return a copy

    def _compute_chord_ssm(self):
        """
        Return the TPS-SSM computed for all chord pairs in the sequence. See
        `segmentation.create_chord_ssm` for parameter specification.
        """
        # Creation and expansion of the TPS-SSM from the harmonic sequence
        chord_ssm, tpsd_cache = seg.create_chord_ssm(
            self._chords, self._keys, normalisation=True,
            as_distance=False, symmetric=True)
        chord_ssm = seg.expand_ssm(chord_ssm, self._times)
        logger.debug(f"TPS-SSM of shape {chord_ssm.shape}")
        return chord_ssm, tpsd_cache

    def _compute_tps_timeseries(self):
        """
        Return the TPS time series associated to this harmonic sequence. The
        type can be either offset- (sequential) or profile- (original) based.
        """
        # Computing the TPS time series and updating class objects
        ts_class = TpsOffsetTimeSeries if self._tpst_type=="offset" \
            else TpsProfileTimeSeries  # parameterise time series type
        tps_timeseries = ts_class(
            self._chords, self._keys, self._times,
            tpsd_cache=self._tpsd_cache)
        return tps_timeseries


class IllegalSegmentationStateException(Exception):
    """Raised when attempting to skip segmentation steps."""
    pass


class HarmonicSegmentation:
    """
    A stateful class for harmonic segmentation, holding results incrementally.
    """
    def __init__(self, harmonic_print:HarmonicPrint) -> None:
        """
        Create a general harmonic segmenter, initialiasing the expected data
        structures to implement a stateful object: boundaries and structures.
        More intermediate states can be set by extending this class.
        """
        self.hprint = harmonic_print
        # Current boundaries detected and additional segmentation output
        self._current_bdect = None  # a list of temporal indexes
        self._current_bdout = None  # a tuple with extra info
        # Structures resulting from the segmentation: boundaries2slices 
        self._harmonic_structures = None  # a list of TpsTimeSeries

    def _flush_segmentation(self):
        self._current_bdect = None
        self._current_bdout = None
        self._harmonic_structures = None

    def segment_harmonic_print(self):
        """
        Returns the segmented harmonic structure following peak detection.

        Returns
        -------
        harmonic_structure : List(tonalpitchspace.TpsTimeSeries)
            The list of harmonic structures as TPS time series.

        Raises
        ------
        IllegalSegmentationStateException
            If attempting to segment before peak detection.

        """
        if self._harmonic_structures is not None:
            return self._harmonic_structures  # use cached version
        if self._current_bdect is None:  # illegal state change
            raise IllegalSegmentationStateException("Detect boundaries first!")
        # Splitting the original time series of the harmonic print
        self._harmonic_structures = self.hprint.tps_timeseries.\
            segment_from_times(self._current_bdect)
        return self._harmonic_structures

    def dump_harmonic_segments(self, out_dir):
        """
        Saves the detected harmonic structures in a pickle file, using the
        identifier of the former chord sequences (e.g. isophonics_0.pickle).
        A list indexing the TPSTimeSeries of the harmonic structures is used.
        """
        if self._harmonic_structures is None:  # illegal state change
            raise IllegalSegmentationStateException("Segmentation required!")

        fpath = os.path.join(out_dir, f"{self.hprint.id}.pkl")
        logger.debug(f"Saving {len(self._harmonic_structures)} in {fpath}")
        with open(fpath, 'wb') as handle:
            pickle.dump(self._harmonic_structures, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def run(self, **segmentation_args):
        """
        Performs all the steps needed to segment the harmonic print.
        """
        raise NotImplementedError("Extend this class and override this method!")


class NoveltyBasedHarmonicSegmentation(HarmonicSegmentation):
    """
    A stateful class for harmonic segmentation, holding results incrementally.
    """
    def __init__(self, harmonic_print: HarmonicPrint) -> None:
        super().__init__(harmonic_print)
        # Novelty detection data structures and parameters
        self._current_novelty = None
        self._current_l_kernel = None
        self._current_var_kernel = None
        # Peak detection algorithm and parameters
        self._pdetection_method = None
        self._pdetection_params = None

    def compute_novelty_curve(self, l: int, var: float, exclude_extremes=True):
        """
        Computes a novelty curve on the harmonic SSM through correlation of
        a Gaussian kernel, of size l*sr and variance `var`, along the main
        diagonal. This value is recomputed if using different parameters.

        Parameters
        ----------
        l : int
            The size, in seconds, of the Gaussian kernel to use.
        var : float
            Variance of the Gaussian kernel to consider.

        Returns
        -------
        novelty_curve : np.ndarray
            The novelty curve computed on the SM using the given parameters.

        """
        if self._current_novelty is not None and \
            self._current_l_kernel == l and self._current_var_kernel == var:
            return self._current_novelty  # use cached version
        # Flush current segmentation results, as not consistent anymore
        self._flush_segmentation()
        # Cache any changed parameter for novelty detection
        self._current_l_kernel, self._current_var_kernel = l, var

        sl = l * self.hprint.sr  # from seconds to samples
        hssm = self.hprint.chord_ssm  # triggers computation 1st time
        self._current_novelty = seg.compute_novelty_sm(
            hssm, L=sl, var=var, exclude=exclude_extremes)

        return self._current_novelty

    def detect_peaks(self, pdetection_method:str, **pdetection_args):
        """
        Performs peak detection on the previously computed novelty curve, using
        the chosen algorithm for peak picking and method-specific parameters.
        Detected peaks are then cached for future calls.

        Parameters
        ----------
        pdetection_method : str
            The name of a supported peak detection algorithm to use; one of 
            segmentation.PDETECTION_MAP`.
        pdetection_args : dict
            A mapping providing method-specific arguments for peak picking. If
            none is provided, the default ones will be used.

        Returns
        -------
        peaks : np.ndarray
            Temporal locations (in samples!) of the detected peaks.
        extra_output : tuple
            Any additional output returned by the peak detection algorithm.

        Raises
        ------
        IllegalSegmentationStateException
            If this step is performed before computing the novelty curve.
        ValueError
            If the requested peak picking method is not supported, yet.
        
        """
        if self._current_bdect is not None and \
            self._pdetection_method == pdetection_method and \
                self._pdetection_params == pdetection_args:
                return self._current_bdect  # use cached version

        if self._current_novelty is None:  # illegal state change
            raise IllegalSegmentationStateException("Compute novelty first!")
        if pdetection_method not in seg.PDETECTION_MAP:
            raise ValueError(f"Not supported algorithm: {pdetection_method}")
        # Flush current segmentation results, and update parameters
        self._flush_segmentation()
        self._pdetection_method, self._pdetection_params = \
            pdetection_method, pdetection_args
        # Perform peak detection with the chosen method and params
        pdetection_fn = seg.PDETECTION_MAP.get(pdetection_method)
        pd_output = pdetection_fn(self._current_novelty, **pdetection_args)
        self._current_bdect, self._current_bdout = pd_output[0], pd_output[1:]

        return self._current_bdect, self._current_bdout


    def run(self, l_kernel, var_kernel, pdetection_method, **pdetection_args):
        """
        Performs all the steps above for harmonic structure analysis.
        """
        self.compute_novelty_curve(l_kernel, var_kernel)
        self.detect_peaks(pdetection_method, **pdetection_args)
        logger.debug(f"Detected peaks at: {self._current_bdect}")
        return self.segment_harmonic_print()


class TimeSeriesHarmonicSegmentation(HarmonicSegmentation):
    """
    A simple harmonic segmentation wrapper for a simple 1-step function.
    """
    def __init__(self, harmonic_print: HarmonicPrint, seg_fn) -> None:
        super().__init__(harmonic_print)
        self.segmentation_fn = seg_fn

    def detect_boundaries(self, **pdetection_args):
        """Apply time series segmentation function using given parameters."""
        self._current_bdect, self._current_bdout = self.segmentation_fn(
            self.hprint.tps_timeseries.time_series, **pdetection_args)

    def run(self, **pdetection_args):
        """Perform time series semantic segmentation and return segments."""
        self.detect_boundaries(**pdetection_args)
        logger.debug(f"Detected boundaries at: {self._current_bdect}")
        return self.segment_harmonic_print()

# ############################################################################ #
# Segmentation baselines as functions
# ############################################################################ #

def split_at_regular_times(time_series, n_regions=2, region_size=None):
    """Return peaks at regular times, using a fixed region size."""
    if n_regions and region_size:
        raise ValueError("Can only specify either number or size of regions!")
    if region_size is None:
        assert n_regions is not None and n_regions > 1
        region_size = int(len(time_series) / n_regions)

    return np.arange(0 , len(time_series), region_size)[1:-1], None


def split_around_regular_times(time_series, n_regions=2, region_size=None, std=None):
    """Return peaks by sampling from a Gaussian centered at regular times."""
    mu_times, _ = split_at_regular_times(
        time_series, n_regions=n_regions, region_size=region_size)
    std = (mu_times[1] - mu_times[0]) / 2 if std is None else std
    return [int(np.random.normal(loc=time, scale=std)) for time in mu_times], _


def split_at_random_times(time_series, n_regions=2):
    """Return peaks at random times withing the given sequence."""
    dur_left = len(time_series) - 1
    current_peaks = [0]

    for i in range(n_regions-1):
        next_dur = np.random.randint(current_peaks[0]+1, dur_left)
        current_peaks.append(current_peaks[-1]+next_dur)
        dur_left = dur_left - next_dur

    return current_peaks[1:], None


def fluss_split(time_series, m, L, n_regions, normalise=True, exc_factor=3):
    """
    Split a time series using the FLUSS algorithm for semantic segmentation.

    Parameters
    ----------
    time_series : np.array
        The time series to segment.
    m : int
        The window size for the computation of the matrix profile.
    L : int
        The subsequence length that is set roughly to be one period length. This
        is likely to be the same value as the window size, m, used to compute
        the matrix profile and matrix profile index but it can be different
        since this is only used to manage edge effects and has no bearing on any
        of the IAC or CAC core calculations.
    n_regions : int
        the number of regimes, n_regimes, to search for (at least 2).
    normalise: bool
        When set to True, this z-normalizes subsequences prior to computing
        distances. Otherwise, this function gets re-routed to its complementary
        non-normalized equivalent in `stumpy`.
    exc_factor : int
        The multiplying factor for the regime exclusion zone. This will nullify
        the beginning and end of the arc curve. Anywhere between 1-5 is
        reasonable according to the paper).

    """
    logger.debug("Computing Matrix Profile")
    mp = stumpy.stump(time_series, m=m, normalize=normalise)
    logger.debug("Starting FLUSS segmentation")
    cac, regime_locations = stumpy.fluss(mp[:, 1],
        L=L, n_regimes=n_regions, excl_factor=exc_factor)

    return sorted(regime_locations), cac


def _rea(cac, n_regimes, L, excl_factor=3):
    """
    Find the location of the regimes using the regime extracting algorithm (REA)
    Taken from https://github.com/TDAmeritrade/stumpy/blob/main/stumpy/floss.py

    Parameters
    ----------
    cac : numpy.ndarray
        The corrected arc curve
    n_regimes : int
        The number of regimes to search for. This is one more than the
        number of regime changes as denoted in the original paper.
    L : int
        The subsequence length that is set roughly to be one period length.
        This is likely to be the same value as the window size, `m`, used
        to compute the matrix profile and matrix profile index but it can
        be different since this is only used to manage edge effects
        and has no bearing on any of the IAC or CAC core calculations.
    excl_factor : int, default 5
        The multiplying factor for the regime exclusion zone
    
    Returns
    -------
    regime_locs : numpy.ndarray
        The locations of the regimes
    Notes
    -----
    DOI: 10.1109/ICDM.2017.21
    <https://www.cs.ucr.edu/~eamonn/Segmentation_ICDM.pdf>`__
    
    """
    regime_locs = np.empty(n_regimes - 1, dtype=np.int64)
    tmp_cac = copy.deepcopy(cac)
    for i in range(n_regimes - 1):
        regime_locs[i] = np.argmin(tmp_cac)
        excl_start = max(regime_locs[i] - excl_factor * L, 0)
        excl_stop = min(regime_locs[i] + excl_factor * L, cac.shape[0])
        tmp_cac[excl_start:excl_stop] = 1.0

    return regime_locs


def floss_split(time_series, m, L, n_regions, normalise=True, exc_factor=1):
    """
    Split a time series using the FLOSS algorithm for semantic segmentation.

    stumpy.floss(mp, T, m, L, excl_factor=5, 

    Parameters
    ----------
    time_series : np.array
        The time series to segment.
    m : int
        The window size for the computation of the matrix profile.
    L : int
        The subsequence length that is set roughly to be one period length. This
        is likely to be the same value as the window size, m, used to compute
        the matrix profile and matrix profile index but it can be different
        since this is only used to manage edge effects and has no bearing on any
        of the IAC or CAC core calculations.
    n_regions : int
        the number of regimes, n_regimes, to search for (at least 2).
    normalise: bool
        When set to True, this z-normalizes subsequences prior to computing
        distances. Otherwise, this function gets re-routed to its complementary
        non-normalized equivalent in `stumpy`.
    exc_factor : int
        The multiplying factor for the regime exclusion zone. This will nullify
        the beginning and end of the arc curve. Anywhere between 1-5 is
        reasonable according to the paper).

    """
    logger.debug("Computing Matrix Profile")
    mp = stumpy.stump(time_series, m=m, normalize=normalise)
    logger.debug("Starting FLOSS segmentation")
    floss_stream = stumpy.floss(
        mp, time_series, m=m,
        L=L, excl_factor=exc_factor)
    regime_locations = _rea(
        floss_stream.cac_1d_, n_regimes=n_regions,
        L=L, excl_factor=exc_factor)

    return sorted(regime_locations), floss_stream.cac_1d_
