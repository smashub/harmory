"""
Script to detect cover songs in a dataset of songs.
"""
import argparse
import logging

import numpy as np
import pandas as pd

from harmory.similarity.compute_similarity import tpsd_similarity, \
    dtw_similarity, longest_common_substring, soft_dtw_similarity
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
               ('dtw', 'offset', 'stretch', 'sakoe_chiba'),
               ('dtw', 'profile', 'stretch', 'sakoe_chiba'),
               ('dtw', 'offset', 'stretch', 'itakura'),
               ('dtw', 'profile', 'stretch', 'itakura'),
               ('dtw', 'offset', 'no-stretch', 'sakoe_chiba'),
               ('dtw', 'profile', 'no-stretch', 'sakoe_chiba'),
               ('dtw', 'offset', 'stretch', 'sakoe_chiba', 'normalize'),
               ('dtw', 'profile', 'stretch', 'sakoe_chiba', 'normalize'),
               ('lcss', 'offset', None, 'sakoe_chiba'),
               ('lcss', 'offset', None, 'itakura'),
               ('sdtw', 'offset', 'stretch'),
               ('sdtw', 'profile', 'stretch'),
               ('sdtw', 'offset', 'stretch', 'sakoe_chiba'),
               ('sdtw', 'profile', 'stretch', 'sakoe_chiba'),
               ('sdtw', 'profile', 'no-stretch', 'sakoe_chiba'),]


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
        self._normalize = None
        self._sakoe_chiba_band = None
        self._itakura_max_slope = None

    def preprocess(self, experiment: tuple) -> None:
        """
        Runs the cover song detection process.
        """
        assert experiment in EXPERIMENTS, f'Invalid experiment: {experiment}'

        logger.info(f'Preprocessing {self._dataset_name} ')

        self._experiment, self._mode = experiment[:2]
        if len(experiment) == 3:
            self._experiment, self._mode, self._stretch = experiment
            assert self._stretch in ['stretch', 'no-stretch', None], \
                f'Invalid stretch mode: {self._stretch}'
        elif len(experiment) == 4:
            self._experiment, self._mode, self._stretch, self._constraint = experiment
            assert self._stretch in ['stretch', 'no-stretch', None], \
                f'Invalid stretch mode: {self._stretch}'
            assert self._constraint in ['sakoe_chiba', 'itakura'], \
                f'Invalid constraint: {self._constraint}'
            self._sakoe_chiba_band = 5 if self._constraint == 'sakoe_chiba' else None
            self._itakura_max_slope = 3 if self._constraint == 'itakura' else None
        elif len(experiment) == 5:
            self._experiment, self._mode, self._stretch, self._constraint, self._normalize = experiment

        self._normalize = True if self._normalize == 'normalize' else False
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
                                        dtw_type=self._experiment,
                                        sakoe_chiba_radius=self._sakoe_chiba_band,
                                        itakura_max_slope=self._itakura_max_slope,
                                        normalize=self._normalize)
        elif self._experiment == 'lcss':
            similarity = longest_common_substring(combinations,
                                                  constraint=self._constraint,
                                                  sakoe_chiba_radius=self._sakoe_chiba_band,
                                                  itakura_max_slope=self._itakura_max_slope, )
        elif self._experiment == 'sdtw':
            similarity = soft_dtw_similarity(combinations,
                                             stretch=self._stretch,
                                             gamma=1,
                                             normalize=self._normalize)
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
    for experiment in EXPERIMENTS[14:]:
        csd.preprocess(experiment)
        similarity = csd.compute_similarity()
        evaluation = csd.evaluate()
        stretch, constraint, normalize = None, None, False
        print(evaluation)
        if len(experiment) == 2:
            distance, tps_mode = experiment
        elif len(experiment) == 3:
            distance, tps_mode, stretch = experiment
        elif len(experiment) == 4:
            distance, tps_mode, stretch, constraint = experiment
        elif len(experiment) == 5:
            distance, tps_mode, stretch, constraint, normalize = experiment
        evaluations.append(['merge',
                            distance,
                            tps_mode,
                            stretch,
                            constraint,
                            normalize,
                            evaluation[0],
                            evaluation[1]])
    print(evaluations)
    save_results(evaluations, '../../exps/results/results_merge_final.csv')

# (0.13713369963369962, 0.23778998778998778)
# (0.7866300366300364, 0.829594017094017)
# (0.7714438339438338, 0.8165445665445664)
# (0.6287393162393162, 0.6891025641025641)
# (0.4791666666666667, 0.5438034188034188)
# (0.71741452991453, 0.7820512820512819)
# (0.6752136752136753, 0.753205128205128)
# (0.7738095238095237, 0.8386752136752136)
# (0.7217643467643465, 0.7914377289377288)
# (0.6602564102564104, 0.7222222222222223)
# (0.5678418803418804, 0.6436965811965809)
# (0.7126068376068376, 0.778846153846154)
# (0.6575854700854701, 0.7628205128205128)
##
# (0.13072344322344323, 0.23084554334554339)
# (0.13713369963369962, 0.23778998778998778)
# (0.6672008547008547, 0.7740384615384613)
# (0.7302350427350429, 0.8103632478632478)
# (0.6672008547008547, 0.7740384615384613)
#
