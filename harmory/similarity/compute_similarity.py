"""
Utility functions for computing similarity between time series.
The functions in this module are of two main types:
1. TPSD based similarity functions
2. DTW based similarity functions
"""
import joblib
import numpy as np
from tslearn.metrics import dtw
from tslearn.preprocessing import TimeSeriesResampler

from harmory.similarity.dataset_processing import process_dataset


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
        print(ts1.time_series)
        print(ts2.time_series)
        longest, shortest = (ts1.time_series, ts2.time_series) if len(
            ts1.time_series) >= len(ts2.time_series) else (ts2.time_series,
                                                           ts1.time_series)
        similarity = minimum_area(list(longest), list(shortest))
        results.append((key1, key2, similarity))
    return results


def dtw_similarity(timeseries_data: np.ndarray,
                   stretch: bool = False) -> list[tuple]:
    """
    Compute the similarity between two time series using the DTW method
    Parameters
    ----------
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
        similarity = dtw(longest, shortest)
        results.append((key1, key2, similarity))
    return results


if __name__ == '__main__':
    # import exps/datasets/cover-song-detection-jams-timeseries/timeseries.pkl
    ts = process_dataset('../../exps/datasets/cover-song-data-jams', save=False)
    permutations = joblib.load(
        '../../exps/datasets/cover-song-data-jams-timeseries/permutations.pkl')
    abc = dtw_similarity(permutations, stretch=True)
