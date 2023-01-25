"""
Core functionalities for similarity and search on harmonic structures.

"""
import logging

import numpy as np

from tslearn.preprocessing import TimeSeriesResampler
from tslearn.neighbors import KNeighborsTimeSeries

logger = logging.getLogger("harmory.similarity")


class HarmonicPatternFinder:

    def __init__(self, model_ckpt=None, dataset=None):
        if model_ckpt is not None:  # model checkpoint available
            self._model = KNeighborsTimeSeries.from_pickle(model_ckpt)
        else:  # remind that a model needs to be created if not provided
            logger.warn("No model checkpoint provided, please create one")
        self._dataset = dataset  # holds the dataset for search using ckpt

    def create_model(self, dataset, num_searches, metric="dtw", n_jobs=1):
        """
        Create and fit a metric-parameterised kNN for time series

        Parameters
        ----------
        dataset : np.ndarray
            The array containing the preprocessed time series.
        num_searches : int
            The number of values to return for each search (k).
        metric : str
            The name of the metric to consider for comparing time series.
        n_jobs : int, optional
            The number of thread that will be used for the search.

        """
        self._model = KNeighborsTimeSeries(
            n_neighbors=num_searches,
            metric=metric, n_jobs=n_jobs)
        self._model.fit(dataset)
        self._dataset = dataset

    def find_similar_patterns(self, query_ts, dist_threshold=None):
        """
        Find harmonic patterns whose distance is below a given threshold.

        Parameters
        ----------
        query : np.ndarray
            A time series that will be used as search query.
        dist_threshold : float
            The maximum distance to retain time series as similar.

        Returns
        -------
        n_simi_ids : list
            A list of time series indexes ordered by shortest distance. E.g. the
            first element `n_simi_dtw[0]` holds the index of the time series
            with the shortest distance from the query, if passing threshold.
        n_simi_dtw : list
            The corresponding list of distances associated to the time series
            indexed by `n_simi_ids`.

        """
        # TODO Check shape of query time series: may need to be stretched
        n_simi_dtw, n_simi_ids = self._model.kneighbors(
            X=[query_ts], return_distance=True)
        # Flatten out the returned arrays: safe for 1 query
        n_simi_dtw, n_simi_ids = n_simi_dtw.ravel(), n_simi_ids.ravel()
        # Discard time series that do not meet the distance threshold
        mask = n_simi_dtw <= dist_threshold
        return list(n_simi_ids[mask]), list(n_simi_dtw[mask])

    def find_similar_patterns_fromidx(self, query_indx, dist_threshold=None):
        """
        Find harmonic patterns for a time series in the dataset given as data
        index; uses the given threshold to filter out distant matches.

        Parameters
        ----------
        query_indx : int
            Index of the time series in the dataset to use as search query.
        dist_threshold : float
            The maximum distance to retain time series as similar.

         Returns
        -------
        simi_indxs : list
            Indexes of the most similar time series in the dataset, where the
            trivial match has been already discarded (`query_indx`).
        simi_dists : list
            The corresponding list of distances associated to the results.

        """
        query_ts = self._dataset[query_indx]  # time series
        simi_indxs, simi_dists = self.find_similar_patterns(
            query_ts, dist_threshold=dist_threshold)
        if query_indx in simi_indxs: # remove the trivial identity match
                trivial_match_idx = simi_indxs.index(query_indx)
                simi_indxs.pop(trivial_match_idx)
                simi_dists.pop(trivial_match_idx)
        # logger.error(f"{query_indx}: {simi_indxs}  {simi_dists}")        
        return simi_indxs, simi_dists


def find_similarities(structure_map, resampling_size, num_searches,
    dist_threshold, metric="dtw", n_jobs=1):

    # structure_ids = np.array(list(structure_map.keys()))  # index-to-ID
    X_data = [tpst.time_series for tpst in structure_map.values()]
    # Computing the average duration of harmonic structures
    hs_durations = [len(x) for x in X_data]
    mu_dur, sigma_dur = np.mean(hs_durations), np.std(hs_durations)
    logger.info(f"Avg. structure dur: {mu_dur:.2f} +/- {sigma_dur:.2f}")

    # Preprocessing of the TPS time series before fitting the model
    X_data = TimeSeriesResampler(sz=resampling_size).fit_transform(X_data)
    logger.debug(f"X_data shape after preprocessing {X_data.shape}")
    # Creation and fitting of the search-based model for harmonic patterns
    hfinder = HarmonicPatternFinder()
    hfinder.create_model(dataset=X_data, num_searches=num_searches,
                         metric=metric, n_jobs=n_jobs)

    hs_simirels, num_processed = [], 0
    ts_simicomp_pool = list(range(len(structure_map)))
    while(len(ts_simicomp_pool) > 0):
        if num_processed % 50 == 0:
            logger.info(f"Processed patterns: {num_processed}")
        current_ts_index = ts_simicomp_pool.pop()  # get next pattern to process
        logger.debug(f"Searching patterns for TS {current_ts_index}")
        # current_ts_id = structure_ids[current_ts_index]  # e.g. isophonics_0_1
        # Find similar harmonic patterns in the dataset, using threshold
        simi_indxs, simi_dists = hfinder.find_similar_patterns_fromidx(
            query_indx=current_ts_index, dist_threshold=dist_threshold)
        # simi_ids = structure_ids[simi_indxs]  # from indexes to IDs
        for match_id, match_dist in zip(simi_indxs, simi_dists):
            relation_type = "sim" # assumed similar as it passed filtration
            if match_dist == 0:  # un-pool identical patterns (distance 0)
                if match_id in ts_simicomp_pool:
                    ts_simicomp_pool.remove(match_id)  # no need to check this
                relation_type = "same"  # declare same pattern for 0 distance
            hs_simirels.append(
                {"source": current_ts_index, "target": match_id,
                "distance": round(match_dist, 2), "type": relation_type})
        num_processed += 1  # update counter for logging

    return hs_simirels, hfinder
