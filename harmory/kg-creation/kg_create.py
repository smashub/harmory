"""
Main script for creating the KG from the file produces in Harmory.
"""
import logging
from pathlib import Path

import rdflib
from rdflib import Graph, Literal, Namespace, RDF
from rdflib.namespace import XSD

from preprocess_kg import PreprocessTrack, PreprocessSimilarity

logger = logging.getLogger('kg-creation.kg_create')

METADATA_PATH = Path('../../data/metadata/meta.csv')

HARMORY = Namespace('http://w3id.org/polifonia/harmory/')
CORE = Namespace('http://w3id.org/polifonia/core/')
MF = Namespace('http://w3id.org/polifonia/musical-features/')


def instantiate_track(graph: rdflib.Graph,
                      dataset_path: str | Path,
                      track_name: str):
    """
    Instantiate a track in the KG.
    Parameters
    ----------
    dataset_path : str | Path
        Path to the directory containing the track data
    graph : rdflib.Graph
        The KG graph to instantiate the track in
    track_name : str
        The name of the track to instantiate
    Returns
    -------

    """
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)
    assert dataset_path.exists(), f"Path {dataset_path} does not exist"
    assert dataset_path.is_dir(), f"Path {dataset_path} is not a directory"

    track = PreprocessTrack(dataset_path / f'{track_name}.pickle')
    track_uri = HARMORY[track_name]
    title, artist, genre = track.get_metadata(METADATA_PATH)

    graph.add((track_uri, RDF.type, CORE.MusicalWork))
    graph.add((track_uri, CORE.hasTitle, Literal(title)))
    graph.add((track_uri, CORE.hasArtist, Literal(artist)))
    if artist:
        graph.add((track_uri, CORE.hasArtist, Literal(artist)))
    if genre:
        graph.add((track_uri, CORE.hasGenre, Literal(genre)))

    for idx, segment in enumerate(track.track_data):
        chords = track.get_chords(idx)
        segment_string = track_name + '$' + '_'.join(chords)
        segment_uri = HARMORY[segment_string]
        durations = track.get_durations(idx)
        graph.add((segment_uri, RDF.type, HARMORY.Segment))
        graph.add(
            (segment_uri, HARMORY.hasOrder, Literal(idx, datatype=XSD.integer)))
        # add next segment
        if idx < len(track.track_data) - 1:
            next_segment = track_name + '$' + '_'.join(
                track.get_chords(idx + 1))
            graph.add(
                (segment_uri, HARMORY.hasNextSegment, HARMORY[next_segment]))
        for chord_idx, chord in enumerate(chords):
            chord_uri = HARMORY[track_name + '$' + chord + '_' + str(chord_idx)]
            start = list(durations)[chord_idx]
            duration = list(durations)[chord_idx + 1] - list(durations)[
                chord_idx]
            graph.add((segment_uri, HARMORY.containsChordAnnotation, chord_uri))
            graph.add((chord_uri, RDF.type, MF.ChordAnnotation))
            graph.add((chord_uri, MF.hasDuration,
                       Literal(duration, datatype=XSD.float)))
            graph.add((chord_uri, MF.hasStartTime,
                       Literal(start, datatype=XSD.float)))
            graph.add((chord_uri, MF.hasChord, Literal(chord)))

        pattern_string = track.get_sequence_string(idx)
        pattern_uri = HARMORY[str(pattern_string)]
        graph.add((track_uri, HARMORY.containsSegmentPattern, pattern_uri))
        graph.add((pattern_uri, RDF.type, HARMORY.SegmentPattern))
        graph.add((pattern_uri, HARMORY.refersToSegment, segment_uri))
        graph.add((segment_uri, HARMORY.hasSegmentPattern, pattern_uri))


def instantiate_similarities(graph: rdflib.Graph,
                             dataset_path: str | Path,
                             map_id_path: str | Path,
                             similarity_path: str | Path):
    """
    Instantiate the similarities in the KG.
    Parameters
    ----------
    dataset_path : str | Path
        Path to the directory containing the track data
    graph : rdflib.Graph
        The KG graph to instantiate the track in
    map_id_path : str | Path
        Path to the file containing the mapping from track to track id
    similarity_path : str | Path
        Path to the directory containing the similarity data
    Returns
    -------

    """
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)
    if isinstance(map_id_path, str):
        map_id_path = Path(map_id_path)
    if isinstance(similarity_path, str):
        similarity_path = Path(similarity_path)
    assert dataset_path.exists(), f"Path {dataset_path} does not exist"
    assert dataset_path.is_dir(), f"Path {dataset_path} is not a directory"
    assert map_id_path.exists(), f"Path {map_id_path} does not exist"
    assert map_id_path.is_file(), f"Path {map_id_path} is not a file"
    assert similarity_path.exists(), f"Path {similarity_path} does not exist"
    assert similarity_path.is_file(), f"Path {similarity_path} is not a file"

    sim = PreprocessSimilarity(dataset_path, map_id_path, similarity_path)

    for identifier in sim.id_map_data:
        sequence = sim.get_sequence_string(identifier)
        sequence_uri = HARMORY[sequence]
        # if sequence in graph.subjects() complement triples
        if sequence_uri in graph.subjects():

            similarities = sim.get_similarities(identifier)
            for similar_sequence in similarities:
                # validate similar_sequence
                if similar_sequence is not None:
                    similar_sequence, similarity_value = similar_sequence
                    similarity_situation = HARMORY[
                        f'{sequence}&{similar_sequence}']
                    graph.add((sequence_uri, HARMORY.isInvolvedInSimilarity,
                               similarity_situation))
                    graph.add((similarity_situation, RDF.type,
                               HARMORY.SegmentPatternSimilarity))
                    graph.add((similarity_situation, HARMORY.hasSimilarityValue,
                               Literal(similarity_value, datatype=XSD.float)))
                    graph.add((similarity_situation,
                               HARMORY.involvesSegmentPattern,
                               HARMORY[similar_sequence]))
                    graph.add((similarity_situation,
                               HARMORY.involvesSegmentPattern, sequence_uri))

def main():
    """
    Main function to create the KG.
    Returns
    -------
    rdflib.Graph
        The KG graph with the track and similarity data
    """
    g = Graph()
    g.bind('har', HARMORY)
    g.bind('core', CORE)
    g.bind('mf', MF)

    # get all files in the dataset directory
    dataset_path = Path('../../data/structures/small-billboard')
    files = [f for f in dataset_path.iterdir() if f.is_file()]
    # instantiate the tracks
    for file in files:
        instantiate_track(g, dataset_path, file.stem)

    # instantiate the similarities
    instantiate_similarities(g,
                             '../../data/structures/small-billboard',
                             '../../data/similarities/pattern2id.pkl',
                             '../../data/similarities/similarities.csv')

    g.serialize(format='turtle', destination='example_complete.ttl')


if __name__ == '__main__':
    g = Graph()
    g.bind('har', HARMORY)
    g.bind('core', CORE)
    instantiate_track(g,
                      '../../data/structures/small-billboard',
                      'billboard_5')

    instantiate_similarities(g,
                             '../../data/structures/small-billboard',
                             '../../data/similarities/pattern2id.pkl',
                             '../../data/similarities/similarities.csv')

    print(g.serialize(format='turtle', destination='example.ttl'))

    main()
