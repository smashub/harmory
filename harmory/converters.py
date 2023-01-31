"""
Utility functions for transforming known sequences into time series.
XXX Taken from `validation/transform_sequences.py` for refactoring.

"""
import pathlib
import logging
from typing import Union
from collections import Counter

import joblib
import numpy as np

from tonalspace import TpsOffsetTimeSeries, TpsProfileTimeSeries

logger = logging.getLogger("harmory.converters")


def convert_sequences(sequences_path: Union[str, pathlib.Path],
                      tpst_type: str = "offset", bundle=False) -> dict[
    str, Union[TpsOffsetTimeSeries, TpsProfileTimeSeries]]:
    """
    Convert known sequences into time series.

    Parameters
    ----------
    sequences_path : str | pathlib.Path
        Path to the directory containing the sequences
    tpst_type : str
        Type of time series to create. Either "offset" or "profile"
    bundle : bool
        Whether sequences should be boundled in a dictionary indexed by sequence
        length (e.g. dict[4] for all sequences of length 4).

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

    with open(sequences_path, "rb") as handle:
        known_sequences = joblib.load(handle)
    ts_class = TpsOffsetTimeSeries if tpst_type == "offset" \
        else TpsProfileTimeSeries

    known_sequences_ts, chord_lens = {}, []
    for sequence_label, (chords, durations, keys) in known_sequences.items():
        if len(chords) == 1:  # check if there is a trivial pattern
            continue  # skip chord 1-gram from TPS projection
        durations = [int(x.replace('*', '1')) for x in durations[:len(chords)]]
        times = np.cumsum([0] + durations)  # from durations to times
        keys = keys[:len(chords)]
        # logger.info(f"Chords: {chords}\nKeys: {keys}\nDur: {durations}")
        assert len(chords) == len(keys) == len(times) - 1
        known_sequences_ts[sequence_label] = \
            ts_class(chords=chords, times=times, keys=keys)
        chord_lens.append(len(chords))  # for stats

    chord_lens_cnt = Counter(chord_lens)
    logger.info(f"Chord length histogram: {chord_lens_cnt}")
    if bundle:  # splitting sequences into separate dict entries
        known_sequences_perlen = {l: {} for l in list(chord_lens_cnt.keys())}
        for name, tps_timeseries in known_sequences_ts.items():
            num_chords = len(tps_timeseries.chords)
            # logger.info(f"Inserting {name} in {ts_len}")
            known_sequences_perlen[num_chords][name] = tps_timeseries
        known_sequences_ts = known_sequences_perlen

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
