"""
Functions for experimenting with similarity measures for chord progressions.
"""
import logging
from pathlib import Path

import jams

from harmory.data import create_chord_sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('harmory.similarity')


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
        logger.info(f'Processing {file}...')
        jam = jams.load(str(file), validate=False, strict=False)
        chords, keys, durations = create_chord_sequence(jam,
                                                        quantisation_unit=1)
        print(chords)


if __name__ == '__main__':
    process_dataset('cover-song-data-jams')
