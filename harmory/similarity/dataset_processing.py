"""
Functions for experimenting with simil measures for chord progressions.
"""
import logging
import os
from pathlib import Path

import joblib
import numpy as np
from itertools import combinations

from harmory.create import HarmonicPrint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('harmory.simil')


# from tpsd.tpsd_comparison import TpsdComparison

def process_dataset(dataset_path: str | Path,
                    tpst_type: str = 'offset',
                    save: bool = False) -> list[tuple[str, np.ndarray]]:
    """
    Process the dataset and save the results as timeseries
    Parameters
    ----------
    dataset_path : str|Path
        Path to the dataset, either as a string or as a Path object
    tpst_type : str, default='offset'
        The type of time series to use, either 'offset' or 'profile'
    save : bool, default=False
        Whether to save the results in a pickle file

    Returns
    -------
    None, saves the results in the dataset folder
    """
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)

    time_series = []

    files = [f for f in dataset_path.iterdir() if f.is_file()]
    for i, file in enumerate(files):
        logger.debug(f'Processing {file}...')
        try:
            sequence = HarmonicPrint(str(file),
                                     sr=1,
                                     chord_namespace='chord_harte',
                                     tpst_type=tpst_type)
            title = sequence.metadata['title']
            title = title.strip()
            time_series.append((str(i) + '$' + title, sequence.tps_timeseries))
        except Exception:
            try:
                sequence = HarmonicPrint(str(file),
                                         sr=1,
                                         chord_namespace='chord',
                                         tpst_type=tpst_type)
                title = sequence.metadata['title']
                title = title.strip()
                time_series.append((str(i) + '$' + title, sequence.tps_timeseries))
            except Exception as e:
                logger.error(f'Error while processing {file}: {e}')
                continue

    if save:
        # create a folder in the parent directory of dataset_path
        ts_path = dataset_path.parent / (dataset_path.name + '-timeseries')
        if not os.path.exists(ts_path):
            os.mkdir(ts_path)
        joblib.dump(time_series, ts_path / f'timeseries_{tpst_type}.pkl')
    return time_series


def get_permutations(time_series: list | str | Path,
                     save: bool = False,
                     output_path: str | Path = None) -> list[
                                                        tuple[tuple, tuple]]:
    """
    Get all the possible permutations of the time series
    Parameters
    ----------
    time_series : dict|str|Path
        The time series to permute, either as a dict, or as a path to a pickle
        file containing the time series
    save : bool, default=False
        Whether to save the results in a pickle file
    output_path : str|Path, default=None
        The path where to save the results, if save is True

    Returns
    -------
    list
        A list of all the possible permutations of the time series
    """
    if isinstance(time_series, str):
        time_series = Path(time_series)
    if isinstance(time_series, Path):
        time_series = joblib.load(time_series)

    permutations = list(combinations(time_series, 2))

    if save:
        assert output_path is not None, "Please provide an output path"
        joblib.dump(permutations, Path(output_path) / 'permutations.pkl')
    return permutations


if __name__ == '__main__':
    ts = process_dataset('../../exps/datasets/cover-song-data-jams', save=True)
    permutations = get_permutations(ts,
                                    save=True,
                                    output_path='../../exps/datasets/cover-song-data-jams-timeseries')
