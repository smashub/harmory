"""
Data manipulations functions.

"""
import re
import logging

import jams
import numpy as np
from music21 import interval, note

from utils import dicted_renaming

logger = logging.getLogger("harmory.data")


def flatten_annotation(listed_ann):
    """
    Flatten a [start, end] LAB-like annotation to retrieve timings only.

    Parameters
    ----------
    listed_ann : list
        A list of LAB-like observations of the format [start, end, value].

    Returns
    -------
    timings : list
        A list of all the distinct temporal boundaries in the annotation.

    """
    timings = set([times for ann in listed_ann for times in ann[:-1]])
    timings = sorted(list(timings))  # flatten and remove duplicates
    assert len(timings) == len(listed_ann) + 1

    return timings


def serialise_jams(jams_object, namespace, annotator=0, lablike=False,
                   merging_delta=1e-3, force_merging=False,
                   force_merging_delta=1):
    """
    Serialise an annotation level of a JAMS object as a list of tuples.

    Parameters
    ----------
    jams_object : jams.JAMS
        A JAMS object from which annotations will be extracted.
    namespace : str
        Name of the namespace to use for retrieving annotations; please, make
        it explicit (e.g. 'chord_harte'), and not general (e.g. 'chord').
    annotator : int
        Which annotation level should be retrieved, if multiple are present.
    lablike : bool
        Whether the annotations will be returned in a LAB-like format, with
        start and end times rather than start and durations.
    merging_delta : float
        Threshold time delta to use for merging consecutive observations when
        the end of the former does not perfectly match the beginning of the
        latter. If their difference is below the delta, times are aligned.

    Notes
    -----
    - Collapse consecutive annotations with the same observation (chord, key).

    """
    ns_annotations = [ann for ann in jams_object.annotations
                      if ann.namespace == namespace]  # exact match
    assert len(ns_annotations) > 0, f"No annotations found for {namespace}"
    ns_annotations = ns_annotations[annotator]  # if more than 1
    observations = [[obs.time, obs.duration, obs.value] \
                    for obs in ns_annotations]

    if lablike:  # make it LAB-like with flat start-end markers
        observations = [[a[0], a[0] + a[1], a[2]] for a in observations]
        for i in range(1, len(observations)):  # attempt local time alignment
            time_delta = observations[i][0] - observations[i - 1][1]
            if time_delta <= merging_delta:  # highly tolerable alignment
                observations[i][0] = observations[i - 1][1]
            elif time_delta <= force_merging_delta and force_merging:
                logger.warn(f"Forcing aligment for {time_delta}s delta")
                observations[i][0] = observations[i - 1][1]
            else:  # this is just to make the user aware of misalignment
                logger.warn(f"Non-consecutive annotation at index {i}")
    return observations


def compensate_key_times(chord_times, key_times, max_offset):
    """
    Attempt to compensate the start and the end of a key annotation to match
    those of a chord annotation. This is done if their temporal offset is below
    a given threshold; an exception is raised otherwise.
    """
    if key_times[0] > chord_times[0]:
        if np.abs(chord_times[0] - key_times[0]) <= max_offset:
            key_times[0] = chord_times[0]  # first key moved earlier
        else:  # difference too large to compensate offset
            raise ValueError("First chord starts with no key")

    if key_times[-1] < chord_times[-1]:
        if np.abs(chord_times[-1] - key_times[-1]) <= max_offset:
            key_times[-1] = chord_times[-1]  # last key held for longer
        else:  # difference too large to compensate offset
            raise ValueError("No key available for last chord")
    elif key_times[-2] < chord_times[-1]:
        key_times[-1] = chord_times[-1]  # safe to shorten last key

    return key_times


def tabularise_annotations(chord_annotations: list, key_annotations: list,
                           max_key_offset: int):
    """
    Tabularise separate LAB-like annotations in a single list.

    Parameters
    ----------
    chord_annotations : list
        A list containing chord observations in LAB-like format.
    key_annotations : list
        A list containing key observations in LAB-like format.
    max_key_offset : int
        The maximum number of quantisation units that is tolerated to align the
        start and the end of a key annotation to a chord sequence. For instance,
        this is needed when the first key starts after the first chord. 

    Returns
    -------
    tabular_annotation : list
        A tabular list where the given annotations are tabularised.

    Notes
    -----
    (*) Generalise this to arbitrary lists of listed annotations.

    """
    def get_frame_element(frame, cnt_element, iterator):
        if frame[0] >= cnt_element[0]:  # frame follows current
            if frame[0] >= cnt_element[1]:  # frame does not meet current
                next_element = next(iterator, None)  # get the next one
                if next_element == None:  # overflow
                    return cnt_element, ""
                #  Go back and check the interval: it might be too early
                return get_frame_element(frame, next_element, iterator)
            assert frame[1] <= cnt_element[1], "Template is not valid"
            return cnt_element, cnt_element[-1]  # frame falls within
        else:  # it is still too early to infill the current element
            return cnt_element, ""  # keep current but park it

    if len(key_annotations) == 1:  # no modulations: just an extra static column
        return [[s, e, c, key_annotations[0][-1]] for s, e, c in
                chord_annotations]
    # Extract and combine timings for annotation alignment
    chord_timings = flatten_annotation(chord_annotations)
    key_timings = flatten_annotation(key_annotations)
    # Compensate start and end times of key annotations and update annotation
    key_timings = compensate_key_times(chord_timings, key_timings, max_key_offset)
    key_annotations[0][0], key_annotations[-1][1] = key_timings[0], key_timings[-1]
    # Concatenate and sort the unique timings of both annotations
    timings = sorted(list(set(chord_timings).union(set(key_timings))))
    # Create a template for each window: implicit global padding
    tabular_annotation = [list(obs) for obs in zip(timings[:-1], timings[1:])]

    chord_iterator = iter(chord_annotations)
    key_iterator = iter(key_annotations)
    cnt_chord = next(chord_iterator)
    cnt_key = next(key_iterator)

    for i in range(len(tabular_annotation)):
        frame = tabular_annotation[i]
        # Get the current chord and key for in-filling the template frame
        cnt_chord, chord = get_frame_element(frame, cnt_chord, chord_iterator)
        cnt_key, key = get_frame_element(frame, cnt_key, key_iterator)
        logger.debug(f"Frame {i} {frame}: key {cnt_key}\tchord {cnt_chord}")
        tabular_annotation[i] = frame + [chord, key]
        logger.debug(f"Tabular {i}: {tabular_annotation[i]}")

    return tabular_annotation


def create_chord_sequence(jam: jams.JAMS, quantisation_unit: float, shift=True,
                          chord_namespace="chord", key_namespace="key_mode"):
    """
    Creates a chord sequence by padding and aligning chord with keys, followed
    by quantisation, left-shift, and removal of empty markers.

    Parameters
    ----------
    jam : jams.JAMS
        The JAMS object containing chord and key annotations.
    quantisation_unit : float
        The unit of quantisation that will be used to discretise times.
    shift : bool
        Whether the sequence will be shifted to remove leading silence, or any
        starting offset encoded in the annotation (e.g. whenever the first
        observation does not start at time 0).

    Returns
    -------
    chords : list
        A list of chords, each occurring in a temporal frame.
    key : list
        A list of keys associated to each chord in the sequence.
    times : list
        A list of quantised onsets for both chord an key annotations.

    Notes
    -----
    - Quantisation should not be applied if the unit is 1, so careful to int.
    - Make sure there are no bubbles otherwise we cannot assume that the start
        of a new chord is the end of the previous!

    """
    # Chords are always assumed to be present in a given JAMS
    chords = serialise_jams(jam, namespace=chord_namespace,
                            lablike=True, merging_delta=5e-1)
    if key_namespace not in [a.namespace for a in jam.annotations]:
        logger.warn("No key annotation in the given JAMS! Using estimation.")
        insert_estimated_key(jam, chords)  # XXX simple method for now
    keys = serialise_jams(jam, namespace=key_namespace, lablike=True,
                          merging_delta=5e-1, force_merging=True,
                          force_merging_delta=3)

    table = tabularise_annotations(chords, keys, max_key_offset=5)
    # Removing trailing no-chord observations from the sequence
    while table[0][2] == "N":  # remove all leading silences
        table.pop(0)
    while table[-1][2] == "N":  # remove all tailing silences
        table.pop(-1)

    # TODO Align key annotations based on delta-threshold: this should remove
    # bubbles with no key/chords and also reduce the size of the chord string;
    #  whenever a buble is found, a no-chord/sil observation N shall be inserted.
    offset = table[0][0] if shift else 0  # amount to pad

    for i in range(len(table)):
        table[i][0] = int((table[i][0] - offset) // quantisation_unit)
        table[i][1] = int((table[i][1] - offset) // quantisation_unit)

    times = list(map(lambda x: x[0], table)) + [table[-1][1]]  # + end
    chords = list(map(lambda x: x[2], table))
    keys = list(map(lambda x: x[3], table))

    return chords, keys, times


def key_to_chord(key: str):
    """
    Generate a simplified chord that represents the tonal centre expressed by
    the given key. This is necessary when computing the TPS profile, a step
    function where the distance of each chord in the progression is computed
    w.r.t. the tonal centre (indicated by the global key).

    Parameters
    ----------
    key : str
        A key, optionally specified along with a mode (e.g. C:min).

    Returns
    -------
    inferred_chord : str
        The simplified chord that was inferred from the given key.

    """
    raise NotImplementedError()


def simplify_harmonic_element(harmo: str):
    """
    Generate a key or chord that simplify the given harmonic element. Although
    the input may be complex, the simplified output that will be inferred by
    this function will only take into account the root and the maj|min quality.

    Parameters
    ----------
    harmo : str
        A chord figure or a key expressed in Harte notation.

    Returns
    -------
    simplification : str
        The simplified key or chord that was inferred from the input.

    """
    # From chord figure in Harte, to a simple key derived from it
    harmo_search = re.search(r"^N|([A-G][b#]?)(:(maj|min))?", harmo)
    simplification = harmo_search.group(1) if harmo_search.group(2) is None \
        else harmo_search.group(1) + harmo_search.group(2)
    return simplification


def insert_estimated_key(jams_object, chords):
    """
    A simple histogram-based method for global key estimation from a JAMS chord
    annotation. Intuitively, the chord that spans the longest time span is used
    as a candidate global key. This methods modifies the JAMS object in-place.
    """
    chordset_duration = {}
    for chord_occurrence in chords:
        chord_figure = chord_occurrence[2]  # e.g. E:min
        current_chordur = chordset_duration.get(chord_figure, 0)
        chordset_duration[chord_figure] =  \
            current_chordur + (chord_occurrence[1] - chord_occurrence[0])

    if "N" in chordset_duration:
        chordset_duration.pop("N")  # we do not want to use N as key
    main_chord = max(chordset_duration, key=chordset_duration.get)
    expected_gkey = simplify_harmonic_element(main_chord)
    expected_end, expected_start = chords[-1][1], chords[0][0]    

    jams_object.annotations.append(jams.Annotation(
        namespace="key_mode", data=[jams.Observation(
            expected_start, expected_end, expected_gkey, confidence=.5)]))


def postprocess_keys(keys, rename_mode={"major":"maj", "minor":"min"}):
    """
    Post-processing operations for key sequences.

    Parameters
    ----------
    keys : list of str
        The list of key to process.
    rename_mode : dict
        A mapping from modes (e.g. 'major') to `key_value` modes (e.g. 'maj').
    
    Returns
    -------
    new_keys : list
        The new key sequence resulting from the required operations.

    """
    new_keys = []
    for key in keys:
        # Apply postprocessing operation in cascade
        new_key = dicted_renaming(key, rename_mode)
        new_keys.append(new_key)
    return new_keys


def postprocess_chords(chords, rename_dict={"X": "N"}, strip_bass=False):
    """
    Post-processing operations on chord sequences.

    Parameters
    ----------
    chords : list
        The list of chords in Harte notation to process.
    rename_dict : dict, optional
        A dictionary for renaming chord figures in Harte.
    strip_bass : bool, optional
        Whether to strip the inversion from the chord.

    Returns
    -------
    new_chords : list
        The new chord sequence resulting from the required operations.

    """
    assert isinstance(chords, list), "Chords must be a list of strings."
    iden = lambda x: x
    #  Declaration of post-processing function as op-or-iden
    rename_fn = (lambda c: rename_dict.get(c, c)) \
        if len(rename_dict) > 0 else iden
    stripbass_fn = (lambda c: c.split("/")[0]) \
        if strip_bass else iden

    new_chords = []
    for chord in chords:
        # Apply postprocessing operation in cascade
        new_chord = rename_fn(chord)
        new_chord = stripbass_fn(new_chord)
        new_chord = convert_harte_bass(new_chord)
        new_chords.append(new_chord)

    return new_chords


def convert_bass_note(bass_note: note.Note,
                      root_note: note.Note,
                      simple: bool = True) -> str:
    """
    Converts a bass note to a chord inversion.

    Parameters
    ----------
    root_note : music21.note.Note
        The root note of the chord.
    bass_note : music21.note.Note
        The bass note of the chord.
    simple : bool

        The mode in which to return the function. If true the interval is
        returned in the music21 "simpleName" mode, in the "name" mode if False.

    Returns
    -------
    inv : str
        The interval between the root and the bass note.

    """
    bass_note.octave = 4
    root_note.octave = 5
    mode = 'simpleName' if simple is True else 'name'
    computed_interval = getattr(interval.Interval(bass_note, root_note), mode)
    return convert_intervals(computed_interval).replace('b2', 'b9').replace('2',
                                                                            '9')


def convert_intervals(m21_interval: str) -> str:
    """
    Utility function that converts intervals from the music21 format to the
    Harte one.
    Parameters
    ----------
    m21_interval : str
        A string containing an interval as expressed by the music21 notation
        (e.g. 'P4').
    Returns
    -------
    harte:interval : str
        A string containing an interval as expressed by the Harte notation
        (e.g. 'b2').
    """
    substitutions = {
        'M': '',
        'm': 'b',
        'P': '',
        'd': 'b',
        'A': '#',
    }

    translation = m21_interval.translate(m21_interval.maketrans(substitutions))
    if m21_interval in ['d2', 'd3', 'd6', 'd7']:
        translation = 'b' + translation
    return translation


def convert_harte_bass(harte_chord: str) -> str:
    """
    Utility function that converts a Harte chord with a wrong annotated bass
    (note instead of interval) to the correct one.
    Parameters
    ----------
    harte_chord : str | Harte
        A string containing a chord in Harte notation as a string

    Returns
    -------
    harte_chord : str
        A string containing a chord in Harte notation with the bass converted
        to an interval.
    """
    if '?' in harte_chord:
        return 'N'
    if '/' in harte_chord and not harte_chord[-1].isdigit():
        base_chord, bass = harte_chord.split('/')
        if ':' not in base_chord:
            base_chord = base_chord + ':maj'
        root, quality = base_chord.split(':')
        bass_interval = convert_bass_note(note.Note(root.replace('b', '-')),
                                          note.Note(bass.replace('b', '-')))
        converted_chord = f'{base_chord}/{bass_interval}'
        return converted_chord
    return harte_chord


if __name__ == '__main__':
    # test utilities
    harte_chord = postprocess_chords(["C#:7/Db", "Gbb:maj", "N", "G:maj/3"])
    print(harte_chord)
