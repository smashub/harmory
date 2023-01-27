"""
Utilities to project and manipulate chord elements in the Tonal Pitch Space.
Some of these utilities may eventually end up in the TPSD project.

See also
    - Lerdahl, Fred. "Tonal pitch space." Music perception (1988): 315-349.
    - https://github.com/andreamust/TPSD
"""
import sys
import logging
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../../TPSD/")  # FIXME after stable version of package
from tpsd.tps_comparison import TpsComparison  #FIXME (see above)

logger = logging.getLogger("tonalspace")


class TpsComparator:
    """
    Cached comparator of chord-key pairs using the Tonal Pitch Space.
    """
    def __init__(self, tpsd_cache=None) -> None:
        if tpsd_cache is not None:
            logger.info("Reusing given TPSD cache for this comparator")
            self._tpsd_cache = tpsd_cache
        else:  # an empty cache will be created otherwise
            logger.info("No cached values provided: creating a new cache")
            self._tpsd_cache = dict()

    def _tps_distance(self, chord_a, key_a, chord_b, key_b):
        # Check if any corner case with silence is found before TPSD
        if chord_a == "N" or chord_b == "N":
            if chord_a != chord_b: return 13  # silence vs chord 
            else: return 0  # both silence has minimum distance
        # No corner cases now, we can safely compute the TPSD
        tpsd_ij = TpsComparison(chord_a=chord_a, key_a=key_a,
                                chord_b=chord_b, key_b=key_b)
        return tpsd_ij.distance()

    def tpsd_lookup(self, chord_a, key_a, chord_b, key_b):
        # First check if head (chord, key) pair is im cache
        head_match = self._tpsd_cache.get((chord_a, key_a), {})
        if (chord_b, key_b) not in head_match:  # check if already cached
            logger.debug(f"Computing new TPS distance for "
                         f"d({chord_a} in {key_a}, {chord_b} in {key_b})")
            head_match[(chord_b, key_b)] = self._tps_distance(
                chord_a=chord_a, key_a=key_a,
                chord_b=chord_b, key_b=key_b)
            self._tpsd_cache[(chord_a, key_a)] = head_match 
        # Safe to return cached value at this stage
        return self._tpsd_cache[(chord_a, key_a)][(chord_b, key_b)]


class TpsTimeSeries:
    """
    Abstract class for harmonic patterns expressed as TPS-based time series.
    """
    def __init__(self, chords, keys, times, sr=1, tpsd_cache=None) -> None:
        # Some sanity checks to make sure that annotations are consistent
        self.durations = np.array(times[1:]) - np.array(times[:-1])
        if not (len(chords) == len(keys) == len(self.durations)):
            raise ValueError("Chords, keys, times size mismatch!")
        # Initialising the TPS comparator with optional cache
        self.tps_comparator = TpsComparator(tpsd_cache)
        self.chords = chords
        self.keys = keys
        self.times = times
        # Data structures that will be created and made available
        self._time_series = self._compute_time_series()

    def _compute_time_series(self):  # real classes should override this
        return NotImplementedError("You should redefine this method")

    @property
    def time_series(self) -> np.ndarray:
        if self._time_series is not None:
            return self._time_series  # return cached version
    
    def plot_time_series(self, quantised=True, figsize=(15,5)):
        """
        Plot the TPS time series, optionally de-quantising temporal units.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.step(range(len(self.time_series)), self.time_series, 'orange')
        ax.set_yticks(np.arange(0, 13 + 1))
        ax.set_title("TPS Time Series")
        ax.set_ylabel('TPS distance')
        # axis.set_xlabel('Seconds after de-quantisation')
        return fig, ax

    def segment_from_times(self, split_times):
        """
        Segment a TPS time series into a number of sub-sequences from the given
        time indexes (t_1, t_2, ... t_n), where t_i denotes the beginning of the
        i-th segment / sub-sequences. More instances of this class are created.

        Parameters
        ----------
        split_times : list
            The onsets when new segments are expected to start. For example,
            [40, 90] will be interpreted as [0, 40, 90, times[-1]] and segment
            the TPS time series in 3 segments: [0, 40], [40, 90], [90, None].
        
        Returns
        -------
        sub_sequences : list[TpsTimeSeries]
            A list containing the new `TpsTimeSeries` objects resulting from the
            segmentation of the original one at the given temporal locations.

        """
        split_times = sorted(split_times)  # make sure timings are consecutive
        if split_times[0] < 0 or split_times[-1] > self.times[-1]:
            raise ValueError("Times are not consistent with this progression")
        # Include extremes times if needed
        if split_times[0] != 0:
            split_times = [0] + split_times
        if split_times[-1] != self.times[-1]:
            split_times += [self.times[-1]]
        # Create temprorary data structures to generate len(split_times) + 1
        # new time series from the current one
        tmp_times, tmp_chords, tmp_keys = \
            deepcopy(self.times), deepcopy(self.chords), deepcopy(self.keys)
        split_indxs = [0] # mirroring split_times for index locations 
        sub_sequences = []  # incrementally holding the new time series
        for i, sptime in enumerate(split_times[1:], 1):
            # Retrieving split location in annotation sequence
            insert_index = np.searchsorted(np.array(tmp_times), sptime)
            split_indxs.append(insert_index)
            logger.debug(f"New segmentation idx for {sptime}: {insert_index}")

            if sptime not in tmp_times: # extending the sequence with new onsets
                tmp_times.insert(insert_index, sptime)
                tmp_chords.insert(insert_index, tmp_chords[insert_index-1])
                tmp_keys.insert(insert_index, tmp_keys[insert_index-1])

            new_tims = tmp_times[split_indxs[i-1]:split_indxs[i]+1]
            new_chos = tmp_chords[split_indxs[i-1]:split_indxs[i]]
            new_keys = tmp_keys[split_indxs[i-1]:split_indxs[i]]
            new_ttts = self.time_series[split_times[i-1]:split_times[i]]
            logger.debug(f"Segmented ts [{split_indxs[i-1]}:{split_indxs[i]}]\n"
                         f"T: {new_tims}\nC: {new_chos}\nK: {new_keys}")
            new_tpstimeseries = self.__class__(
                chords=new_chos, keys=new_keys, times=new_tims,
                tpsd_cache=self.tps_comparator._tpsd_cache)
            new_tpstimeseries._time_series = new_ttts  # safe here
            sub_sequences.append(new_tpstimeseries)

        return sub_sequences


class TpsOffsetTimeSeries(TpsTimeSeries):
    """
    Return the TPS offset time series, where each element in the curve
    represents the TPS distance of the current chord with respect to the
    previous one in time (depending on the quantisation/sampling level).
    """
    # overrides
    def _compute_time_series(self) -> np.ndarray:
        super()  # this will return the cached version, if previously computed
        # Adding a fake chord:key pair for simplifying the loop below
        chords = [self.keys[0]] + self.chords
        keys = [self.keys[0]] + self.keys
        tps_offsets = []  # incrementally holds chord offset distances
        for i in range(1, len(chords)):
            # Compute the TPS distance w.r.t. the previous chord
            offset = self.tps_comparator.tpsd_lookup(
                chord_a=chords[i], key_a=keys[i],
                chord_b=chords[i-1], key_b=keys[i-1])
            tps_offsets = tps_offsets + [offset]*self.durations[i-1]

        self._time_series = np.array(tps_offsets)
        return self._time_series


class TpsProfileTimeSeries(TpsTimeSeries):
    """
    Return the TPS profile, where the distance of each chord at each time step
    is computed against the global key of the piece. The latter, is assumed to
    be found as the first key in the annotation provided at construction time.
    """
    # overrides
    def _compute_time_series(self) -> np.ndarray:
        super()  # this will return the cached version, if previously computed
        global_key = self.keys[0]  # first key assumed to be the global one
        tps_profile = []  # incrementally holds TPS profile step by step
        for i in range(len(self.chords)):
            # Compute the TPS distance w.r.t. the global key
            hvar = self.tps_comparator.tpsd_lookup(
                chord_a=self.chords[i], key_a=self.keys[i],
                chord_b=global_key, key_b=global_key)
            tps_profile = tps_profile + [hvar]*self.durations[i]

        self._time_series = np.array(tps_profile)
        return self._time_series
