"""
Utilities to compute statistical properties from structures and similarities.
"""
# Average neighbour per node (may correspond to node degree in the graph)
# Average distance in the neighbour per structure/node
# Average number of same nodes?
# Proportion of unique patterns that do not appear anywhere else
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from tslearn.preprocessing import TimeSeriesResampler
from tslearn.neighbors import KNeighborsTimeSeries

from search import HarmonicPatternFinder

logger = logging.getLogger("harmory.analysis")


def compute_structure_statistics(hstructures_map):
    """
    Computes the following descriptors from the segmentation:
    - Number of structures per harmonic progression;
    - Average duration of structrures per harmonic progression;
    - Overall number of structures (global);
    - Average duration of structures (global).
    """
    # Computing global descriptors first
    X_data = [tpst.time_series for tpst in hstructures_map.values()]
    # Computing the average duration of harmonic structures
    hs_durations = [len(x) for x in X_data]  # FIXME check sampling rate <- this assumes sr=1
    glob_dur_mu, glob_dur_std = np.mean(hs_durations), np.std(hs_durations)
    logger.info(f"Global duration: {glob_dur_mu:.2f} +/- {glob_dur_std:.2f}")
    glob_num_structures = len(hstructures_map)
    # Computing structure-aggregated descriptors
    num_structures_php, dur_structures_php = {}, {}
    for structure_id, structure_tpsts in tqdm(hstructures_map.items()):
        # Retrieving ChoCo ID from Structure ID
        choco_id = "_".join(structure_id.split("_")[:-1])
        # Updating the number of structures per harmonic progression
        num_structures_php[choco_id] = num_structures_php.get(choco_id, 0) + 1
        # Updating the duration of structures per harmonic progression
        dur_structures_php[choco_id] = dur_structures_php.get(choco_id, []) \
            + [len(structure_tpsts.time_series)] # add current duration

    dur_structures_php = [np.mean(durs) for durs in dur_structures_php.values()]
    num_structures_php = list(num_structures_php.values())

    return {
        "glob_num_structures": glob_num_structures,
        "glob_durations": hs_durations,
        "glob_dur_mu": glob_dur_mu,
        "glob_dur_std": glob_dur_std,
        "num_structures_php": num_structures_php,
        "dur_structures_php": dur_structures_php
    }


def find_pattern_groups(simi_relations, all_pattern_ids):
    """
    Partition pattern IDs into three lists, representing their types: pattern
    families (those denoting a group of patterns that are found repeated),
    pattern friendly (those that are not found repeated, but still similar to
    otehers); and unique or orphan or unused patterns (those that are never 
    found repeated nor similar to other patterns). 

    Parameters
    ----------
    simi_relations : pd.DataFrame
        The pandas `DataFrame` listing all the similarity relationships betweem
        patterns (with distinction between `similar` and `same` types).
    all_pattern_ids : list or nd.array
        A list, or array, containing the IDs of all patterns that were found.

    Returns
    -------
    pattern_families : list
        A list of the IDs denoting the pattern families.
    pattern_friendly : list
        A list containing the IDs of those patterns that are only found similar.
    pattern_unique : list
        The IDs of those patterns that are never found repeated nor similar.

    """
    # Used patterns are those patterns with at least 1 similarity relationship
    used_patterns = list(set(simi_relations["source"].unique())\
        .union(set(simi_relations["target"].unique())))
    # A pattern family wrap a group of patterns which are considered the same
    pattern_families = list(simi_relations[
        simi_relations["type"]=="same"]["source"].unique())
    pattern_friendly = [p for p in used_patterns if p not in pattern_families]
    # Unused patterns are all the rest: those that were not found elsewhere
    pattern_unique = [p for p in all_pattern_ids if p not in used_patterns]

    return pattern_families, pattern_friendly, pattern_unique


def get_patterns_in_family(pattern_family_id, simi_relations):
    """Returns the list of pattern IDs belonging to the same pattern family."""
    return list(simi_relations[(simi_relations["source"]==pattern_family_id)
                                & (simi_relations["type"]=="same")]["targets"])


def get_patterns_in_neighbourood(pattern_id, simi_relations, include_same=True):
    """Returns the list of pattern IDs that are similar to the given one."""
    neighbours = simi_relations[simi_relations["source"]==pattern_id]
    if not include_same:  # discard 'same' patterns, if required
        neighbours = neighbours[neighbours["type"]=="simi"]
    return list(neighbours["target"]), list(neighbours["distance"]) 


def compute_similarity_statistics(simi_relations_df, pattern_ids):
    """
    Computes a set of descriptors from the similarity relations found, which are
    returned in a single dictionary indexed by metric name.
    """
    simi_relations = simi_relations_df[simi_relations_df["type"]=="sim"]
    no_simi_rels = len(simi_relations)
    no_same_rels = np.abs(len(simi_relations_df) - no_simi_rels)
    simi_rel_avg =  np.mean(simi_relations["distance"])
    simi_rel_std =  np.std(simi_relations["distance"])

    # simi_relations_df = pd.read_csv(similarities_path)
    # Extracting groups of patterns from their relation types
    pattern_families, pattern_friendly, pattern_unique = \
        find_pattern_groups(simi_relations_df, pattern_ids)

    pattern_neigh_sizes = []  # size of neighbourhood regardless of type
    pattern_family_sizes, pattern_simila_sizes = [], []  # more granular here
    pattern_avg_distance_all, pattern_avg_distance_nosame = [], []  # distances

    for pattern_id in pattern_friendly + pattern_families:
        _, neighbour_dist = get_patterns_in_neighbourood(
            pattern_id, simi_relations_df)
        neighbour_dist = np.array(neighbour_dist)
        # Extracting neighbourhood and family-wise statistics
        pattern_neigh_sizes.append(len(neighbour_dist))
        pattern_simila_sizes.append(len(neighbour_dist))
        if pattern_id in pattern_families:
            pattern_family_sizes.append(np.sum(neighbour_dist==0))
            pattern_simila_sizes[-1] = pattern_neigh_sizes[-1] \
                                       - pattern_family_sizes[-1]
        # Now computing the average distance of similar patterns 
        pattern_avg_distance_all.append(np.mean(neighbour_dist))
        pattern_avg_distance_nosame.append(
            np.mean(neighbour_dist[neighbour_dist>0]))

    return {
        "no_simi_rels": no_simi_rels,
        "no_same_rels": no_same_rels,
        "simi_rel_avg": simi_rel_avg,
        "simi_rel_std": simi_rel_std,
        "no_pattern_families": len(pattern_families),
        "no_pattern_friendly": len(pattern_friendly),
        "no_pattern_unique": len(pattern_unique),
        "pattern_neigh_sizes": pattern_neigh_sizes,
        "pattern_family_sizes": pattern_family_sizes,
        "pattern_simila_sizes": pattern_simila_sizes,
        "pattern_avg_distance_all": pattern_avg_distance_all,
        "pattern_avg_distance_nosame": pattern_avg_distance_nosame,
    }


def create_timeseries_dataset(tpstimeseries_dict: dict, resampling_size: int):
    # structure_ids = np.array(list(structure_map.keys()))  # index-to-ID
    X_data = [tpst.time_series for tpst in tpstimeseries_dict.values()]
    # Preprocessing of the TPS time series before fitting the model
    X_data = TimeSeriesResampler(sz=resampling_size).fit_transform(X_data)
    logger.debug(f"X_data shape after preprocessing: {X_data.shape}")


class PatternValidationFinder(HarmonicPatternFinder):

    def create_model(self, dataset: str, resampling_size: int, num_tophits: int,
                     metric_name="dtw", metric_params=None, n_jobs=1):
        """
        Create and fit a metric-parameterised kNN for time series.

        Parameters
        ----------
        dataset : list
            A list of `TpsTimeSeries` objects to create a time series dataset.
        resampling_size : int
            Length of the resulting time series after resampling operations.
        num_tophits : int
            The number of known harmonic patterns that will be returned after
            the search as top hits for a query. If > 1, all hits are averaged.
        metric_name : str
            The name of the metric to consider for comparing time series.
        metric_params : dict
            A dictionary holding metric-specific parameters.
        n_jobs : int, optional
            The number of thread that will be used for the search.

        Notes
        -----
        (*) Parameterise the standardisation/normalisation of time series

        """
        # From a dataset of TpsTimeSeries to a general time series dataset.
        # This includes time stretching of time series to a common length.
        khpatterns = create_timeseries_dataset(dataset, resampling_size)
        # Creation of the model from the given parameters
        self._model = KNeighborsTimeSeries(
            n_neighbors=num_tophits, n_jobs=n_jobs,
            metric=metric_name, metric_params=metric_params)
        self._model.fit(dataset)
        self._dataset = khpatterns

    def check_pattern_presence(self, patterns):
        pass
        # Perform kNN search using patterns as query against khpatterns
        n_simi_dtw, n_simi_ids = self._model.kneighbors(
            X=[patterns], return_distance=True)
        # Returng the max for each vector distance; saving (khpattern_i, dist)

        # Return a list of tuples containing the ID of the closest known harmonic
        # pattern and the corresponding distance for each query pattern.

        # TODO Check shape of query time series: may need to be stretched
        
        
        return list(n_simi_ids[mask]), list(n_simi_dtw[mask])
