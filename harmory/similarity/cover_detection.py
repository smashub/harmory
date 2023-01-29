"""
Script to detect cover songs in a dataset of songs.
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd

from harmory.similarity.compute_similarity import tpsd_similarity, \
    dtw_similarity
from harmory.similarity.dataset_processing import process_dataset, \
    get_permutations
from harmory.similarity.evaluation import get_covers, covers_ranking, \
    evaluate

logger = logging.getLogger('harmory.similarity.cover_detection')

EXPERIMENTS = [('tpsd', 'offset'),
               ('tpsd', 'profile'),
               ('dtw', 'offset', 'stretch'),
               ('dtw', 'profile', 'stretch'),
               ('dtw', 'offset', 'no-stretch'),
               ('dtw', 'profile', 'no-stretch'),
               ('ctw', 'offset', 'stretch'),
               ('ctw', 'profile', 'stretch'),
               ('ctw', 'offset', 'no-stretch'),
               ('ctw', 'profile', 'no-stretch'),]


class CoverSongDetection:
    """
    Class to detect cover songs in a dataset of songs.
    """

    def __init__(self, dataset_path: str, n_jobs: int = 1):
        """
        Constructor for the CoverSongDetection class.
        Parameters
        ----------
        dataset_path : str
            Path to the dataset
        n_jobs : int, default=1
            Number of jobs to run in parallel
        """
        self._dataset_path = dataset_path
        self._n_jobs = n_jobs
        self._dataset_name = dataset_path.split('/')[-1]
        self._cache = {}
        self._experiment = None
        self._mode = None
        self._stretch = None
        self._similarity = None
        self._constraint = None

    def preprocess(self, experiment: tuple) -> None:
        """
        Runs the cover song detection process.
        """
        assert experiment in EXPERIMENTS, f'Invalid experiment: {experiment}'

        logger.info(f'Preprocessing {self._dataset_name} ')

        if len(experiment) == 2:
            self._experiment, self._mode = experiment
        elif len(experiment) == 3:
            self._experiment, self._mode, self._stretch = experiment
            assert self._stretch in ['stretch', 'no-stretch'], \
                f'Invalid stretch mode: {self._stretch}'
        elif len(experiment) == 4:
            self._experiment, self._mode, self._stretch, self._constraint = experiment
            assert self._stretch in ['stretch', 'no-stretch'], \
                f'Invalid stretch mode: {self._stretch}'
            assert self._constraint in ['sakoe_chiba', 'itakura'], \
                f'Invalid constraint: {self._constraint}'
        else:
            raise ValueError('Invalid experiment')

        self._stretch = True if self._stretch == 'stretch' else False

        logging.info(f'Preprocessing {self._dataset_name} '
                     f'using {experiment} and {self._mode} mode')
        if f'{self._dataset_name}_{self._mode}' not in self._cache.keys():
            time_series = process_dataset(self._dataset_path,
                                          tpst_type=self._mode,
                                          save=False)
            self._cache[f'{self._dataset_name}_{self._mode}'] = time_series
        else:
            time_series = self._cache[f'{self._dataset_name}_{self._mode}']

        if f'{self._dataset_name}_{self._mode}_combinations' not in self._cache.keys():
            combinations = get_permutations(time_series)
            self._cache[
                f'{self._dataset_name}_{self._mode}_combinations'] = combinations
        else:
            pass

    def compute_similarity(self) -> list[tuple]:
        """
        Computes the similarity between two time series.
        :param ts1: the first time series
        :type ts1: np.ndarray
        :param ts2: the second time series
        :type ts2: np.ndarray
        :return: the similarity between the two time series
        :rtype: float
        """
        assert self._experiment is not None, 'Experiment not set'
        assert self._mode is not None, 'Mode not set'
        if self._experiment == 'dtw' or self._experiment == 'ctw':
            assert self._stretch is not None, 'Stretch not set'

        logger.info(f'Computing similarity for {self._dataset_name} using '
                    f'{self._experiment} and {self._mode} mode')

        combinations = self._cache[
            f'{self._dataset_name}_{self._mode}_combinations']

        if self._experiment == 'tpsd':
            similarity = tpsd_similarity(combinations)
        elif self._experiment == 'dtw' or self._experiment == 'ctw':
            similarity = dtw_similarity(combinations,
                                        stretch=self._stretch,
                                        constraint=self._constraint,
                                        dtw_type=self._experiment)
        else:
            raise ValueError(f'Invalid experiment: {self._experiment}')

        self._similarity = similarity

        return self._similarity

    def evaluate(self):
        """
        Evaluates the cover song detection process.
        """
        logger.info(f'Evaluating {self._dataset_name} using '
                    f'{self._experiment} and {self._mode} mode')

        time_series = self._cache[f'{self._dataset_name}_{self._mode}']
        covers = get_covers(time_series)
        print(covers)

        ranking = covers_ranking(self._similarity)

        return evaluate(covers, ranking)


def save_results(results: list[list], output_path: str) -> None:
    """
    Saves the results of the cover song detection process.
    :param results: the results of the cover song detection process
    :type results: list[tuple]
    :param output_path: the path to the output directory
    :type output_path: str
    :return: None
    :rtype: None
    """
    results = pd.DataFrame(results, columns=['dataset', 'distance',
                                             'tps_mode', 'stretch',
                                             'cinstraint', 'first_tier',
                                             'second_tier'])
    results.to_csv(output_path, index=False)


def main():
    """
    Main function for the cover song detection script.
    Returns
    -------
    None.
    """
    parser = argparse.ArgumentParser(description='Cover song detection script')

    parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('output_path', type=str,
                        help='Path to the output directory')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of jobs to be used')

    args = parser.parse_args()

    evaluations = []
    csd = CoverSongDetection(args.dataset_path, args.n_jobs)
    for experiment in EXPERIMENTS:
        csd.preprocess(experiment)
        csd.compute_similarity()
        evaluation = csd.evaluate()
        if len(experiment) == 2:
            distance, tps_mode = experiment
            stretch = None
            constraint = None
        elif len(experiment) == 3:
            distance, tps_mode, stretch = experiment
            constraint = None
        elif len(experiment) == 4:
            distance, tps_mode, stretch, constraint = experiment
        evaluations.append([(args.dataset_path.split('/')[-1],
                            distance,
                            tps_mode,
                            stretch,
                            constraint,
                            evaluation[0],
                            evaluation[1])])
    save_results(evaluations, args.output_path)

    return evaluations


if __name__ == '__main__':
    csd = CoverSongDetection('../../exps/datasets/merge',
                             n_jobs=1)
    evaluations = []
    for experiment in EXPERIMENTS:
        csd.preprocess(experiment)
        similarity = csd.compute_similarity()
        evaluation = csd.evaluate()
        if len(experiment) == 2:
            distance, tps_mode = experiment
            stretch = None
            constraint = None
        elif len(experiment) == 3:
            distance, tps_mode, stretch = experiment
            constraint = None
        elif len(experiment) == 4:
            distance, tps_mode, stretch, constraint = experiment
        evaluations.append(['merge',
                            distance,
                            tps_mode,
                            stretch,
                            constraint,
                            evaluation[0],
                            evaluation[1]])
    print(evaluations)
    save_results(evaluations, '../../exps/results/results_merge.csv')
