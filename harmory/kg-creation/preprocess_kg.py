"""
Utility functions for preprocessing track data and similarity data for the
creation of a knowledge graph (KG).
"""
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname('../../harmory')))


class PreprocessData:
    """
    Class for preprocessing track data and similarity data for the creation of
    the Harmory KG.
    """

    def __init__(self,
                 file_path: str | Path,
                 id_map_path: str | Path,
                 similarity_path: str | Path):
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
        if isinstance(id_map_path, str):
            id_map_path = Path(id_map_path)
        if isinstance(similarity_path, str):
            similarity_path = Path(similarity_path)
        assert file_path.exists(), f"Path {file_path} does not exist"
        assert file_path.is_file(), f"Path {file_path} is not a file"
        assert id_map_path.exists(), f"Path {id_map_path} does not exist"
        assert id_map_path.is_file(), f"Path {id_map_path} is not a file"
        assert similarity_path.exists(), f"Path {similarity_path} does not exist"
        assert similarity_path.is_file(), f"Path {similarity_path} is not a file"

        self._file_path = file_path
        self._id_map_path = id_map_path
        self._file_name = file_path.stem
        self._similarity_path = similarity_path

        self.track_data = joblib.load(self._file_path)
        id_map_data = joblib.load(self._id_map_path)
        self.id_map_data = {v: k for k, v in id_map_data.items()}
        self.similarity_data = pd.read_csv(self._similarity_path, sep=',')

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

    def match_identifier(self):
        """
        Match the track id to the track data.
        Returns
        -------

        """
        similarities = []
        for idx, segment in enumerate(self.track_data):
            # get the segments in the similarity file
            assert f'{self._file_name}_{idx}' in self.id_map_data.keys(), \
                f"Track id for {self._file_name}_{idx} not in map data"
            segment_id = self.id_map_data[f'{self._file_name}_{idx}']
            if segment_id in self.similarity_data['source'].values:
                line = self.similarity_data.loc[
                    (self.similarity_data['source'] == segment_id) & (
                            self.similarity_data['type'] != 'same')]
                sim = (line['target'], line['distance'])
                if sim not in similarities:
                    similarities.append(sim)
            elif segment_id in self.similarity_data['target'].values:
                # check if pointer is in target and the type is "same"
                line = self.similarity_data.loc[
                    (self.similarity_data['target'] == segment_id) & (
                            self.similarity_data['type'] != 'same')]
                sim = (line['source'], line['distance'])
                if sim not in similarities:
                    similarities.append(sim)
            else:
                pass

        return similarities


if __name__ == '__main__':
    # abc = joblib.load('../../data/structures/small-audio/billboard_12.pkl')
    # print(abc[0].time_series)
    #
    # cde = joblib.load('../../data/similarities/pattern2id.pkl')
    # print(cde)

    pre = PreprocessData(
        '../../data/structures/small-billboard/billboard_5.pickle',
        '../../data/similarities/pattern2id.pkl',
        '../../data/similarities/similarities.csv')
    pre.get_metadata('../../data/metadata/meta.csv')
    print(pre.get_sequence_string(3))
