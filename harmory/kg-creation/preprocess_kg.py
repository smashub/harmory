"""
Utility functions for preprocessing track data and similarity data for the
creation of a knowledge graph (KG).
"""
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname('../../harmory')))


class PreprocessTrack:
    """
    Class for preprocessing track data and similarity data for the creation of
    the Harmory KG.
    """

    def __init__(self,
                 file_path: str | Path):
        """
        Constructor for PreprocessData class.
        Parameters
        ----------
        file_path : str | Path
            Path to the directory containing the track data
        id_map_path : str | Path
            Path to the file containing the mapping from track to track id
        """
        self._genre = None
        self._artist = None
        self._title = None
        if isinstance(file_path, str):
            file_path = Path(file_path)
        assert file_path.exists(), f"Path {file_path} does not exist"
        assert file_path.is_file(), f"Path {file_path} is not a file"

        self._file_path = file_path
        self._file_name = file_path.stem

        self.track_data = joblib.load(self._file_path)

    def get_metadata(self, metadata_file_path: str | Path) -> tuple:
        """
        Get metadata from the track data.
        Parameters
        ----------
        metadata_file_path : str | Path
            Path to the directory containing the metadata
        Returns
        -------
        tuple
            Tuple of metadata
        """
        if isinstance(metadata_file_path, str):
            metadata_file_path = Path(metadata_file_path)
        assert metadata_file_path.exists(), f"Path {metadata_file_path} does not exist"
        assert metadata_file_path.is_file(), f"Path {metadata_file_path} is not a file"

        # import csv file with pandas
        metadata = pd.read_csv(metadata_file_path, sep=',', header=None)
        # get row for self._file_name
        metadata = metadata.loc[metadata[0] == self._file_name]

        if len(metadata) == 0:
            raise ValueError(f"Metadata for {self._file_name} not found")

        self._title = metadata[1].values[0] if not pd.isna(
            metadata.iloc[0, 1]) else None
        self._artist = metadata[2].values[0] if not pd.isna(
            metadata.iloc[0, 2]) else None
        self._genre = metadata[7].values[0] if not pd.isna(
            metadata.iloc[0, 7]) else None

        return self._title, self._artist, self._genre

    def get_sequence(self, sequence_idx: int) -> list:
        """
        Get the sequence of segments for a given track.
        Parameters
        ----------
        sequence_idx : int
            Index of the sequence
        Returns
        -------
        list
            List of segments
        """
        if sequence_idx < 0 or sequence_idx >= len(self.track_data):
            raise ValueError(f"Index {sequence_idx} out of range")
        return self.track_data[sequence_idx].time_series

    def get_sequence_string(self, sequence_idx: int) -> str:
        """
        Get the sequence of segments for a given track.
        Parameters
        ----------
        sequence_idx : int
            Index of the sequence
        Returns
        -------
        list
            List of segments
        """
        if sequence_idx < 0 or sequence_idx >= len(self.track_data):
            raise ValueError(f"Index {sequence_idx} out of range")
        return '_'.join(
            [str(x) for x in self.track_data[sequence_idx].time_series])


class PreprocessSimilarity:
    """
    Class to preprocess the similarity data.
    """

    def __init__(self,
                 dataset_path: str | Path,
                 id_map_path: str | Path,
                 similarity_path: str | Path):
        """
        Constructor for the PreprocessSimilarity class.
        Parameters
        ----------
        id_map_path : str | Path
            Path to the id map file
        similarity_path : str | Path
            Path to the similarity file
        """
        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)
        if isinstance(id_map_path, str):
            id_map_path = Path(id_map_path)
        if isinstance(similarity_path, str):
            similarity_path = Path(similarity_path)

        assert dataset_path.exists(), f"Path {dataset_path} does not exist"
        assert dataset_path.is_dir(), f"Path {dataset_path} is not a directory"
        assert id_map_path.exists(), f"Path {id_map_path} does not exist"
        assert id_map_path.is_file(), f"Path {id_map_path} is not a file"
        assert similarity_path.exists(), f"Path {similarity_path} does not exist"
        assert similarity_path.is_file(), f"Path {similarity_path} is not a file"

        self._dataset_path = dataset_path
        self._id_map_path = id_map_path
        self._similarity_path = similarity_path

        self.id_map_data = joblib.load(self._id_map_path)
        # self.id_map_data = {v: k for k, v in id_map_data.items()}
        self.similarity_data = pd.read_csv(self._similarity_path, sep=',')
        self._file_names = [x.stem for x in self._dataset_path.glob('*')]

    def get_sequence(self, pattern_id: int) -> np.ndarray:
        """
        Get the sequence of segments for a given track.
        Parameters
        ----------
        pattern_id : int
            Index of the sequence to search
        Returns
        -------
        list
            Time series of the segment
        """
        assert pattern_id in self.id_map_data.keys(), \
            f"Pattern id {pattern_id} not in map data"
        file = self.id_map_data[pattern_id].split('_')
        file_name, sequence_idx = '_'.join(file[:-1]), int(file[-1])
        assert file_name in self._file_names, \
            f"File name {file_name} not in file names"
        track_data = joblib.load(self._dataset_path / f'{file_name}.pickle')

        return track_data[sequence_idx].time_series

    def get_sequence_string(self, pattern_id: int) -> str:
        """
        Get the sequence of segments for a given track.
        Parameters
        ----------
        pattern_id : int
            Index of the sequence to search
        Returns
        -------
        list
            Time series of the segment
        """
        return '_'.join([str(x) for x in self.get_sequence(pattern_id)])

    def get_similarities(self, pattern_id: int) -> list[tuple[int, float]]:
        """
        Get the similarities for a given track.
        Parameters
        ----------
        pattern_id : int
            Index of the sequence to search
        Returns
        -------
        list[tuple[int, float]]
            List of tuples with the pattern id and the distance
        """
        assert pattern_id in self.id_map_data.keys(), \
            f"Pattern id {pattern_id} not in map data"
        similarities = []
        sources = self.similarity_data.loc[
            self.similarity_data['source'] == pattern_id]
        targets = self.similarity_data.loc[
            self.similarity_data['target'] == pattern_id]
        filtered = pd.concat([sources, targets])

        for line in filtered.itertuples():
            sim = (line.target, line.distance)
            if sim not in similarities:
                similarities.append(sim)

        return similarities


if __name__ == '__main__':
    # abc = joblib.load('../../data/structures/small-audio/billboard_12.pkl')
    # print(abc[0].time_series)
    #
    # cde = joblib.load('../../data/similarities/pattern2id.pkl')
    # print(cde)

    pre = PreprocessTrack(
        '../../data/structures/small-billboard/billboard_5.pickle')
    pre.get_metadata('../../data/metadata/meta.csv')
    print(pre.get_sequence_string(3))

    sim = PreprocessSimilarity('../../data/structures/small-billboard',
                               '../../data/similarities/pattern2id.pkl',
                               '../../data/similarities/similarities.csv')
    print(sim.get_similarities(1457))
