"""
Simple utilities for the sonification of chord sequences in Harte notation.
"""
import os
import time
import logging

import note_seq
import numpy as np
import soundfile as sf

import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

from harte.harte import Harte

logger = logging.getLogger("sonification")


def save_notesequence(note_sequence, out_dir, midi_fname=None, suffix=""):
    """
    Save the given note-sequence as a MIDI file in the desired directory.
    """
    if midi_fname is None:  # use a default naming convention
        date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
        midi_fname = '%s_%s.mid' % (date_and_time, str(suffix))
    note_seq.sequence_proto_to_midi_file(
        note_sequence, os.path.join(out_dir, midi_fname))


def synthesise_sequence(note_sequence, out_file=None,
    synth=note_seq.midi_synth.synthesize,
    sample_rate=44100, **synth_args):
    """
    Sonification of a note-sequence object, through a synthetizer.

    Parameters
    ----------
    sequence : music_pb2.NoteSequence
        A notesequence that will be synthesized for playback.
    out_file : str
        A filename of the audio file to be written.
    synth : fn
        A synthesis function that takes a sequence and sample rate as input.
    sample_rate : int
        The sample rate at which to synthesize.
    synth_args : dict
        Additional keyword arguments to pass to the synth function.

    Return
    ------
    wave : np.ndarray
        The waveform generated from the synthesised note sequence.

    """
    wave = synth(note_sequence, sample_rate=sample_rate, **synth_args)
    if out_file is not None:  # write the audio file to disk, if required
        sf.write(out_file, wave, sample_rate)

    return wave


def get_harmonic_notesequence(chords:list, times:list, fix_times=False, prog=0,
                              shift=False):
    """
    Create a note-sequence encoding the given harmonic progression.

    Parameters
    ----------
    chords : list
        A list of chords labels encoded in Harte notation.
    times : list
        The timing information of the chord sequence, where times(i) denotes the
        onset of chords(i), and `len(times) == len(chords) + 1` for the latter.
    fix_times : bool
        Wether to ignore chord timings and use fixed durations for chords.
    prog : int
        MIDI program number that will be associated to the chord stream.
    shift : bool
        Whether to remove any initial silence that prevents the sequence from
        starting at time 0.

    Returns
    -------
    chord_ns : music_pb2.NoteSequence
        The generated `NoteSequence` object for the given chord annotations.

    """
    if times is not None and len(times) != len(chords) + 1:
        raise ValueError("Invalid times vector, chord alignment expected!")
    if fix_times:  # use fixed times and ignore actual times, if provided
        logger.warning("Using fixed temporal spans for chord timings!")
        times = list(range(len(chords) + 1))
    if shift:  # times are left-shifted to make the sequence start at 0
        sequence_start_time = times[0]
        times = [t - sequence_start_time for t in times]

    chord_ns = note_seq.protobuf.music_pb2.NoteSequence()
    for i, chord_fig in enumerate(chords):  # iterate over all chords
        start_time, end_time = times[i], times[i+1]
        # Remove bass note before parsing the chord
        # chord_nob = strip_chord_bass(chord_fig)  # FIXME
        chord = Harte(chord_fig)  # from chord string to M21
        chord_pitches = [pitch.midi for pitch in chord.pitches]
        for pitch in chord_pitches:  # add each note constituent
            chord_ns.notes.add(pitch=pitch, velocity=80, program=prog,
                               start_time=start_time, end_time=end_time)

    chord_ns.total_time = chord_ns.notes[-1].end_time
    chord_ns.tempos.add(qpm=120)

    return chord_ns


def sonify_chord_sequence(chord_annotations, track_name, soundfont,
  out_dir="./", fix_times=False, prog=0):
  """
  Notes: the entry point should be a JAMS file.
  """
  chord_ns = get_harmonic_notesequence(
    chord_annotations, fix_times=fix_times, prog=prog)

  save_notesequence(
    chord_ns, os.path.join(out_dir, "midi"), f"{track_name}.mid")
  synthesise_sequence(
    chord_ns, os.path.join(out_dir, "audio", f"{track_name}.wav"),
    synth=note_seq.midi_synth.fluidsynth, sf2_path=soundfont)

