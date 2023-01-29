"""
Scripts for unwrapping annotations in the CASD dataset.
"""
import os
from pathlib import Path

import jams


def unwrap_casd(dataset_path: str, output_path) -> None:
    """
    Utility function to unwrap the annotations in the CASD dataset.
    Parameters
    ----------
    dataset_path : str
        The path to the dataset
    output_path : str
        The path to the output directory

    Returns
    -------
    None
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    jams_files = list(Path(dataset_path).glob('**/*.jams'))
    for jam_file in jams_files:
        jam = jams.load(str(jam_file))
        key_annotation = jam.annotations.search(namespace='key')[0]
        for idx, annotation in enumerate(jam.annotations):
            if annotation.namespace == 'chord':
                new_jam = jams.JAMS()
                new_jam.file_metadata = jam.file_metadata
                new_jam.sandbox = jam.sandbox
                new_jam.annotations.append(annotation)
                new_jam.annotations.append(key_annotation)
                file_path = str(Path(output_path) / f'{jam_file.stem}_{idx}.jams')
                new_jam.save(file_path)


if __name__ == '__main__':
    unwrap_casd('../../exps/datasets/CASD', '../../exps/datasets/CASD-unwrap')
