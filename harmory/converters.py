"""
Utility functions for transforming known sequences into time series.
XXX Taken from `validation/transform_sequences.py` for refactoring.

"""
import pathlib
from typing import Union

import joblib

from tonalspace import TpsOffsetTimeSeries, TpsProfileTimeSeries


def convert_sequences(sequences_path: Union[str, pathlib.Path],
                      tpst_type: str = "offset") -> dict[
    str, Union[TpsOffsetTimeSeries, TpsProfileTimeSeries]]:
    """
    Convert known sequences into time series.

    Parameters
    ----------
    sequences_path : str | pathlib.Path
        Path to the directory containing the sequences
    tpst_type : str
        Type of time series to create. Either "offset" or "profile"

    Returns
    -------
    dict[str, TpsOffsetTimeSeries | TpsProfileTimeSeries]
        Dictionary of time series, indexed by sequence label
    """
    if isinstance(sequences_path, str):
        sequences_path = pathlib.Path(sequences_path)
    assert sequences_path.exists(), f"Path {sequences_path} does not exist"
    assert sequences_path.is_file(), f"Path {sequences_path} is not a file"
    assert tpst_type in ["offset",
                         "profile"], f"Unknown time series type {tpst_type}"

    known_sequences = joblib.load(sequences_path)
    ts_class = TpsOffsetTimeSeries if tpst_type == "offset" \
        else TpsProfileTimeSeries

    known_sequences_ts = {}
    for sequence_label, (chords, durations, keys) in known_sequences.items():
        durations = [float(x.replace('*', '1')) for x in
                     durations[:len(chords)]]
        durations = durations + [durations[0]]
        keys = keys[:len(chords)]
        print(f"Chords: {chords}\nKeys: {keys}\nDur: {durations}")
        assert len(chords) == len(keys) == len(durations) - 1
        known_sequences_ts[sequence_label] = ts_class(chords=chords,
                                                      times=durations,
                                                      keys=keys)

    return known_sequences_ts


def save_known_sequences(sequences_path: Union[str, pathlib.Path],
                         output_path: Union[str, pathlib.Path],
                         save_mode: type = Union[list, dict],
                         tpst_type: str = "offset") -> None:
    """
    Save known sequences as time series.

    Parameters
    ----------
    sequences_path : str | pathlib.Path
        Path to the directory containing the sequences
    output_path : str | pathlib.Path
        Path to the directory where the time series should be saved
    save_mode : type (list | dict)
        Whether to save the time series as a list or a dictionary. The list
        will lose the sequence labels, while the dictionary will keep them.
    tpst_type : str
        Type of time series to create. Either "offset" or "profile"

    Returns
    -------
    None, saves the time series to disk to the specified path
    """
    assert save_mode in [list, dict], f"Unknown save mode {save_mode}"

    known_sequences_ts = convert_sequences(sequences_path, tpst_type)
    output_path = pathlib.Path(output_path) / f"known_sequences_ts.pkl"

    if save_mode == list:
        data = list(known_sequences_ts.values())
    elif save_mode == dict:
        data = known_sequences_ts
    joblib.dump(data, output_path)
