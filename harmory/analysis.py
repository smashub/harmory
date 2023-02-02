"""
Utilities to compute statistical properties from structures and similarities.
"""
# Average neighbour per node (may correspond to node degree in the graph)
# Average distance in the neighbour per structure/node
# Average number of same nodes?
# Proportion of unique patterns that do not appear anywhere else
import os
import logging
import argparse
from typing import Union

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from tslearn.preprocessing import TimeSeriesResampler
from tslearn.neighbors import KNeighborsTimeSeries

import constants
from search import HarmonicPatternFinder
from harmseg import load_structures_nested
from utils import set_logger, get_directories, create_dir

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


def create_timeseries_dataset(tpstimeseries, resampling_size: int):
    if isinstance(tpstimeseries, dict):
        tpstimeseries = list(tpstimeseries.values())
    X_data = [tpst.time_series for tpst in tpstimeseries]
    # TODO Standardisation of time series should happen here
    # Preprocessing of the TPS time series before fitting the model
    X_data = TimeSeriesResampler(sz=resampling_size).fit_transform(X_data)
    logger.debug(f"X_data shape after preprocessing: {X_data.shape}")
    return X_data


class PatternValidationFinder(HarmonicPatternFinder):

    #  self._resampling_size

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
        (*) TODO Parameterise the standardisation/normalisation of time series

        """
        # From a dataset of TpsTimeSeries to a general time series dataset.
        # This includes time stretching of time series to a common length.
        khpatterns = create_timeseries_dataset(dataset, resampling_size)
        # Creation of the model from the given parameters
        self._model = KNeighborsTimeSeries(
            n_neighbors=num_tophits, n_jobs=n_jobs,
            metric=metric_name, metric_params=metric_params)
        self._model.fit(khpatterns)
        self._dataset = khpatterns
        self._resampling_size = resampling_size

    def check_pattern_presence(self, query_patterns):
        # Perform kNN search using patterns as query against khpatterns
        pattern_dataset = create_timeseries_dataset(
            query_patterns, self._resampling_size)
        n_simi_dist, n_simi_ids = self._model.kneighbors(
            X=pattern_dataset, return_distance=True)
        # Return the min for each vector distance; saving (khpattern_i, dist)
        # To generate: a list of tuples containing the ID of the closest known
        # harmonic pattern and the corresponding distance for each query.
        return (n_simi_dist, n_simi_ids) if n_simi_dist.shape[1] > 1 \
            else (n_simi_dist.ravel(), n_simi_ids.ravel())


def find_known_patterns(segments, track_id, hp_validator):
    """
    Use the the given PatternValidationFinder to retrieve ID and distance of
    the most similar known pattern handled by the finder -- for each segemnt.
    """
    validation_records = []
    # Retrieve the best match/pattern for each segmented structure
    top_dists, top_ids = hp_validator.check_pattern_presence(segments)
    for i, (top_dist, top_id) in enumerate(zip(top_dists, top_ids)):
        validation_records.append([track_id, i, top_dist, top_id])
    # Some syntactic sugar here: saving min and mean for each segment
    validation_records.append([track_id, -1, np.min(top_dists),
                            top_ids[np.argmin(top_dists)]])
    validation_records.append([track_id, -2, np.mean(top_dists),
                            len(set(top_ids))])
    # Track-specific validation results
    return validation_records


def measure_segmentation_coverage(
    structures_dir: str, known_patterns: Union[str, dict], resampling_size: int, 
    split=None, metric_name="dtw", metric_params=None, n_jobs=1):
    """
    Evaluates a harmonic segmentation against a collection of known patterns.

    Parameters
    ----------
    structures_dir : str
        Path to the directory with the output of the harmonic segmnentation.
    known_patterns : Union[str, dict]
        Path to the dump containing the TpsTimeSeries of known patterns. A
        dictionary containing the latter, indexed by name, can also be provided.
    resampling_size : int
        The size of time series for enabling time invariant pattern search.
    split : _type_, optional
        Name of the split with the known patterns to use in this experiment. It
        may correspond, for instance, to the length of known harmonic patterns.
    metric_name : str, optional
        Name of the distance metric used to compare time series with each other.
    metric_params : dict, optional
        Parameters of the metric specified before; use defaults otherwise.
    n_jobs : int, optional
        Number of threads for parallel exection; use -1 for all.

    Returns
    -------
    validation_df : pd.DataFrame
        A pandas Dataframe containg the pattern coverage, per segment. It also
        contains aggregated statistics, distinguished by -1 (min) and -2 (mean).
    
    """
    if isinstance(known_patterns, str):
        with open(known_patterns, "rb") as handle:
            known_patterns = joblib.load(handle)
    if split is not None:  # use only a partition/split of all patterns
        if split not in set(known_patterns.keys()):  # check split name
            raise ValueError(f"Split {split} is not a valid key!")
        logger.info(f"Using known patterns of length {split} --- "
                    f"Found {len(known_patterns[split])} patterns.")
        known_patterns = known_patterns[split]

    # Create the pattern validator for known patterns
    hp_validator = PatternValidationFinder()
    hp_validator.create_model(known_patterns, resampling_size, num_tophits=1, 
        metric_name=metric_name, metric_params=metric_params, n_jobs=n_jobs)

    hstructures_per_track = load_structures_nested(structures_dir)
    logger.info(f"Loaded {len(hstructures_per_track)} segmentations")
    # FIXME Each segmentation is managed by an available thread here
    # Parallel support does not work here yet, probably for parallel access
    # validation_records =  Parallel(n_jobs=n_jobs)(delayed(find_known_patterns)\
    #             (segments, track_id, hp_validator) for track_id, segments \
    #                 in tqdm(hstructures_per_track.items()))
    validation_records = []
    for track_id, segments in tqdm(hstructures_per_track.items()):
        # logger.info(f"Validating segmentation for {track_id}")
        validation_records += find_known_patterns(
            segments, track_id, hp_validator)

    validation_df = pd.DataFrame(validation_records,
        columns=["choco_id", "segment", "top_dist", "top_pattern"])
    validation_df['top_pattern'] = validation_df['top_pattern'].astype('int')
    if split is not None:  # append extra column for split name
        validation_df["split"] = split

    return validation_df


def measure_segmentation_coverage_per_split(
    structures_dir: str, known_patterns: Union[str, dict], resampling_size: int, 
    exclude_splits=[2], metric_name="dtw", metric_params=None, n_jobs=1,
    parallelise_splits=False):
    """
    Computes patterns coverage of segments for all splits. For more info, see
    `measure_segmentation_coverage`.
    """
    with open(known_patterns, "rb") as handle:
        known_patterns = joblib.load(handle)
    splits = list(known_patterns.keys())
    if exclude_splits is not None:  # remove unwanted splits
        splits = [s for s in splits if s not in exclude_splits]

    logger.info(f"Running coverage evaluation for {splits} splits")
    if not parallelise_splits:  # sequential version: better to debug
        validation_dfs = []
        for split in splits:  # process each split, then merge
            segmentation_split_coverage = measure_segmentation_coverage(
                split=split,
                structures_dir=structures_dir,
                known_patterns=known_patterns,
                resampling_size=resampling_size,
                metric_name=metric_name,
                metric_params=metric_params,
                n_jobs=n_jobs)
            validation_dfs.append(segmentation_split_coverage)
    else:  # Running parallel version across splits
        raise NotImplementedError()

    final_validation_df = pd.concat(validation_dfs)
    final_validation_df.reset_index(drop=True, inplace=True)
    return final_validation_df


def summarise_segmentation_coverage_per_split(validation_df: pd.DataFrame):
    """
    Aggregate validation results with respect to different splits.
    """
    min_agg_df = validation_df[validation_df["segment"]==-1]
    mean_agg_df = validation_df[validation_df["segment"]==-2]
    min_agg_df = min_agg_df[["top_dist", "split"]]
    mean_agg_df = mean_agg_df[["top_dist", "split"]]
    # Aggregating results per split, using sequence-wise results
    mean_agg_mean_df = mean_agg_df.groupby("split").mean()
    mean_agg_std_df = mean_agg_df.groupby("split").std()
    min_agg_mean_df = min_agg_df.groupby("split").mean()
    min_agg_std_df = min_agg_df.groupby("split").std()
    # Merging all the result sets within the same dataframe
    merged_agg_df = pd.concat([
        mean_agg_mean_df, mean_agg_std_df,
        min_agg_mean_df, min_agg_std_df], axis=1)
    merged_agg_df.columns = ["mean_mean_dist", "std_mean_dist", 
                            "mean_min_dist", "std_min_dist"]
    merged_agg_df.reset_index(inplace=True)  # keep index column
    return merged_agg_df


def measure_segmentation_coverage_from_grid(grid_dir, known_patterns,
    resampling_size, metric_name="dtw", metric_params=None, n_jobs=1):
    """
    Evaluate a segmentation method run on a parameter grid (1 method, N runs).
    """
    method = os.path.basename(grid_dir)
    configuration_names = get_directories(grid_dir)
    logger.info(f"Found {len(configuration_names)} runs for {method}")
    grid_validation_dfs = []
    for configuration_str in tqdm(configuration_names):  # loops over runs
        logger.info(f"Evaluating {method} - {configuration_str}")
        # Retrieving the actual path of this run, with structures inside
        run_dir = os.path.join(grid_dir, configuration_str)
        # Measuing coverage of known patterns for all possible lengths
        validation_df = measure_segmentation_coverage_per_split(
            run_dir, known_patterns=known_patterns,
            metric_name=metric_name, metric_params=metric_params,
            resampling_size=resampling_size, exclude_splits=[2], n_jobs=n_jobs)
        # Save evaluation results, aggregate per split, append name of paramset
        validation_df.to_csv(os.path.join(run_dir, "coverage.csv"), index=False)
        validation_df = summarise_segmentation_coverage_per_split(validation_df)
        validation_df["params"] = configuration_str
        grid_validation_dfs.append(validation_df)
    # Time to combine all results in a single dataframe
    grid_validation_dfs = pd.concat(grid_validation_dfs)
    grid_validation_dfs.reset_index(drop=True, inplace=True)
    grid_validation_dfs["method"] = method
    # Saving to disk and bye bye
    out_fname = os.path.join(grid_dir, "coverage_all.csv")
    logger.info(f"Writing final grid results for {method} at: {out_fname}")
    grid_validation_dfs.to_csv(out_fname, index=False)


def main():

    COMMANDS = ["similarities", "segmentation", "segwrite"]

    parser = argparse.ArgumentParser(
        description="Analysis of Harmory segmentation and similarities.")
    parser.add_argument('cmd', type=str, choices=COMMANDS,
                        help=f"Either {', '.join(COMMANDS)}.")

    parser.add_argument('data', type=str,
                        help='Directory with segmentations, or similarities.')

    parser.add_argument('--known_patterns', type=str,
                        help='Path to the pickle file with known patterns.')
    parser.add_argument('--out_dir', type=str,
                        help='Directory where all output will be saved.')
    parser.add_argument('--metric_name', type=str, default="dtw",
                        help='Name of the distance metric for time series.')
    parser.add_argument('--n_workers', action='store', type=int, default=1,
                        help='Number of workers for stats computation.')
    parser.add_argument('--resampling_size', action='store', type=int, default=30,
                        help='Size of time series after resampling for comp.')
    parser.add_argument('--debug', action='store_true',
                        help='Whether to run in debug mode: slow and logging.')
    parser.add_argument('--log_level', action='store', default=logging.INFO,
                        help='Whether to run in debug mode: slow and logging.')

    args = parser.parse_args()
    set_logger("harmory", args.log_level)

    if args.out_dir is None:
        args.out_dir = args.data
    create_dir(args.out_dir)

    dtw_config = dict(global_constraint = constants.DTW_GLOBAL_CONSTRAINT,
                      sakoe_chiba_radius = constants.DTW_SAKOE_RADIUS)

    if args.cmd == "segmentation":
        print(f"SEGMENT: Segmenting chord sequences into harmonic structures")
        measure_segmentation_coverage_from_grid(
            grid_dir=args.data,
            known_patterns=args.known_patterns,
            resampling_size=args.resampling_size,
            metric_name=args.metric_name,
            metric_params=dtw_config,
            n_jobs=args.n_workers)

    elif args.cmd == "segwrite":
        methods = get_directories(args.data)
        print(f"SEGWRITE: Writing merged segementation results of: {methods}")

        segmentation_res_df = []
        for method in methods:  # attempt to retrieve all result sets
            method_df_loc = os.path.join(args.data, method, "coverage_all.csv")
            if os.path.isfile(method_df_loc):
                logger.info(f"Reading results from {method_df_loc}")
                segmentation_res_df.append(pd.read_csv(method_df_loc))
        # Aggregation of results in a single dataframe
        segmentation_res_df = pd.concat(segmentation_res_df)
        segmentation_res_df.reset_index(drop=True, inplace=True)
        # Ordering data for inspection and dumping results to disk
        fname = os.path.join(args.out_dir, "tmp_sample_res.csv")
        logger.info(f"Writing merged output in {fname}")
        segmentation_res_df.sort_values(
            ["split","mean_mean_dist", "mean_min_dist"])\
                .to_csv(fname, index=None)

    else:  # trivially, args.cmd == "similarities"
        raise NotImplementedError()

    print("DONE!")


if __name__ == "__main__":
    main()
