"""
Functions for experimenting with simil measures for chord progressions.
"""
import logging
from pathlib import Path

import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..')))

from harmory.create import HarmonicPrint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('harmory.simil')


# from tpsd.tpsd_comparison import TpsdComparison

def process_dataset(dataset_path: str | Path) -> None:
    """
    Process the dataset and save the results as timeseries
    Parameters
    ----------
    dataset_path : str|Path
        Path to the dataset, either as a string or as a Path object

    Returns
    -------
    None, saves the results in the dataset folder
    """
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)

    # read all files in the dataset
    files = [f for f in dataset_path.iterdir() if f.is_file()]
    for file in files:
        logger.debug(f'Processing {file}...')
        sequence = HarmonicPrint(str(file), sr=1, chord_namespace='chord_harte')
        title = sequence.metadata['title']
        title = title.strip()
        print(title)


if __name__ == '__main__':
    process_dataset('cover-song-data-jams')
