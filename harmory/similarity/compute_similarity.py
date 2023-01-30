"""
Utility functions for computing similarity between time series.
The functions in this module are of two main types:
1. TPSD based similarity functions
2. DTW based similarity functions
"""
import numpy as np
from tslearn.metrics import dtw, ctw, lcss, soft_dtw
from tslearn.preprocessing import TimeSeriesResampler, \
    TimeSeriesScalerMeanVariance

from harmory.similarity.dataset_processing import process_dataset, \
    get_permutations


def minimum_area(longest_sequence, shortest_sequence) -> float:
    """
    Calculates the minimum area between the step functions calculated over
    two chord sequences
    :return: a floating number which corresponds to the value of minimum
    area between the two areas
    """
    minimum_area = 0
    longest = longest_sequence[:]
    for step in range(
            len(longest_sequence) - len(shortest_sequence) + 1):
        if step != 0:
            longest.pop(0)
        area = 0
        for beat in range(len(shortest_sequence)):
            lower = shortest_sequence[beat] if shortest_sequence[
                                                   beat] <= longest[
                                                   beat] else \
                longest[beat]
            higher = shortest_sequence[beat] if shortest_sequence[
                                                    beat] > longest[
                                                    beat] else \
                longest[beat]

            area += higher - lower
        if minimum_area > area or step == 0:
            minimum_area = area
    return minimum_area / len(shortest_sequence)


def tpsd_similarity(timeseries_data: np.ndarray) -> list[tuple]:
    """
    Compute the similarity between two time series using the TPSD method
    Parameters
    ----------
    timeseries_data : np.ndarray
        Array containing the time series data to be compared

    Returns
    -------
    dict
        Dictionary containing the similarity score for each pair of time series
    """
    results = []
    for pair in timeseries_data:
        (key1, ts1), (key2, ts2) = pair
        # transform to lists
        ts1, ts2 = list(ts1.time_series), list(ts2.time_series)
        longest, shortest = (ts1, ts2) if len(ts1) >= len(ts2) else (ts2, ts1)
        similarity = minimum_area(longest, shortest)
        results.append((key1, key2, similarity))
    return results


def dtw_similarity(timeseries_data: np.ndarray,
                   dtw_type: str = 'dtw',
                   stretch: bool = False,
                   constraint: str = None,
                   sakoe_chiba_radius: int = None,
                   itakura_max_slope: int = None,
                   normalize: bool = False) -> list[tuple]:
    """
    Compute the similarity between two time series using the DTW method
    Parameters
    ----------
    normalize
    itakura_max_slope
    sakoe_chiba_radius
    constraint
    dtw_type
    timeseries_data : np.ndarray
        Array containing the time series data to be compared
    stretch : bool, optional
        Whether to stretch the shortest time series to match the length of the
        longest one, by default False

    Returns
    -------
    dict
        Dictionary containing the similarity score for each pair of time series
    """
    results = []
    for pair in timeseries_data:
        (key1, ts1), (key2, ts2) = pair
        longest, shortest = (ts1.time_series, ts2.time_series) if len(
            ts1.time_series) >= len(ts2.time_series) else (ts2.time_series,
                                                           ts1.time_series)
        if stretch:
            shortest = TimeSeriesResampler(sz=len(longest)).fit_transform(
                shortest)[0]
        if normalize:
            longest, shortest = longest.reshape(1, -1), shortest.reshape(1, -1)
            longest = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(
                longest)
            longest = longest[0].ravel()
            shortest = TimeSeriesScalerMeanVariance(mu=0.,
                                                    std=1.).fit_transform(
                shortest)
            shortest = shortest[0].ravel()

        assert dtw_type in ['dtw', 'ctw'], 'Invalid dtw type!'
        if dtw_type == 'dtw':
            similarity = dtw(longest, shortest,
                             global_constraint=constraint,
                             sakoe_chiba_radius=sakoe_chiba_radius,
                             itakura_max_slope=itakura_max_slope)
        if dtw_type == 'ctw':
            similarity = ctw(longest, shortest,
                             global_constraint=constraint,
                             sakoe_chiba_radius=sakoe_chiba_radius,
                             itakura_max_slope=itakura_max_slope)
        results.append((key1, key2, similarity))
    return results


def longest_common_substring(timeseries_data: np.ndarray,
                             constraint: str = None,
                             sakoe_chiba_radius: int = None,
                             itakura_max_slope: int = None
                             ) -> list[tuple]:
    """
    Compute the similarity between two time series using the longest common
    substring method
    Parameters
    ----------
    itakura_max_slope
    sakoe_chiba_radius
    constraint
    timeseries_data : np.ndarray
        Array containing the time series data to be compared

    Returns
    -------
    dict
        Dictionary containing the similarity score for each pair of time series
    """
    results = []
    for pair in timeseries_data:
        (key1, ts1), (key2, ts2) = pair
        longest, shortest = (ts1.time_series, ts2.time_series) if len(
            ts1.time_series) >= len(ts2.time_series) else (ts2.time_series,
                                                           ts1.time_series)
        similarity = lcss(shortest, longest, eps=1,
                          global_constraint=constraint,
                          sakoe_chiba_radius=sakoe_chiba_radius,
                          itakura_max_slope=itakura_max_slope)
        results.append((key1, key2, 1 - similarity))
    return results


def soft_dtw_similarity(timeseries_data: np.ndarray,
                        stretch: bool = False,
                        gamma: float = .1,
                        normalize: bool = False) -> list[tuple]:
    """
    Compute the similarity between two time series using the Soft DTW method
    Parameters
    ----------
    normalize
    stretch
    timeseries_data : np.ndarray
        Array containing the time series data to be compared
    gamma : float, optional
        Softening parameter, by default 1.0

    Returns
    -------
    dict
        Dictionary containing the similarity score for each pair of time series
    """
    results = []
    for pair in timeseries_data:
        (key1, ts1), (key2, ts2) = pair
        longest, shortest = (ts1.time_series, ts2.time_series) if len(
            ts1.time_series) >= len(ts2.time_series) else (ts2.time_series,
                                                           ts1.time_series)
        if stretch:
            shortest = TimeSeriesResampler(sz=len(longest)).fit_transform(
                shortest)[0]
        if normalize:
            longest, shortest = longest.reshape(1, -1), shortest.reshape(1, -1)
            longest = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(
                longest)
            longest = longest[0].ravel()
            shortest = TimeSeriesScalerMeanVariance(mu=0.,
                                                    std=1.).fit_transform(
                shortest)
            shortest = shortest[0].ravel()
        similarity = soft_dtw(longest, shortest, gamma=gamma)
        results.append((key1, key2, similarity))
    return results


if __name__ == '__main__':
    # import exps/datasets/cover-song-detection-jams-timeseries/timeseries.pkl
    ts = process_dataset('../../exps/datasets/cover-song-data-jams', save=False)
    permutations = get_permutations(ts)
    # abc = longest_common_substring(permutations, constraint='sakoe_chiba', sakoe_chiba_radius=5)
    abc = soft_dtw_similarity(permutations, stretch=True, gamma=1.0)
