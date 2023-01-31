"""
Main entry point for recreating the Harmonic Memory from scratch, or extend the
memory with new harmonic progressions or patterns manually provided as inputs.

"""
import os
import pickle
import logging
import argparse

from collections import OrderedDict

import jams
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


from config import ConfigFactory
from search import find_similarities
from harmseg import HarmonicPrint, NoveltyBasedHarmonicSegmentation
from utils import get_files, get_filename, set_logger, create_dir

logger = logging.getLogger("harmory.create")


def create_segmentation(jams_path, config, out_dir):

    hprint = HarmonicPrint(jams_path,
        sr=config["sr"], tpst_type=config["tpst_type"])
    hprint.run()  # creates SSM and TPS time series
    segmenter = NoveltyBasedHarmonicSegmentation(hprint)

    segmenter.run(
        l_kernel=config["l_kernel"],
        var_kernel=config["var_kernel"],
        pdetection_method=config["pdetection_method"],
        **config["pdetection_params"])

    segmenter.dump_harmonic_segments(out_dir)


def load_structures(structures_dir):
    """
    Read all harmonic structures previously dumped in separate files and merge
    them in a dictionary indexed by ChoCo/song ID and segment index.
    """
    structures_map = OrderedDict()
    structures_pkls = get_files(structures_dir, "pkl", full_path=True)
    logger.info(f"Found {len(structures_pkls)} dumps in {structures_dir}")

    for structure_pkl in tqdm(structures_pkls):
        # Retrieve ChoCo ID from file name and read all structures
        choco_id = get_filename(structure_pkl, strip_ext=True)
        with open(structure_pkl, 'rb') as handle:
            hstructures = pickle.load(handle)
        # Adding the harmonic structure to the main map
        for i, hstructure in enumerate(hstructures):
            hstructure_id = f"{choco_id}_{i}"
            structures_map[hstructure_id] = hstructure

    logger.info(f"Found {len(structures_map)} harmonic structures")
    return structures_map


def create_similarities(structures_dir, config, out_dir, n_jobs=1):
    """
    Find similarities between harmonic patterns as time series data. As output,
    the following data structures will be saved to disk:
    - `hfinder.pkl`, holding a checkpoint of the model used for search;
    - `similarities.csv`, a pandas edgelist with similarity relationships;
    - `pattern2id.pkl`, a mapping from pattern indexes to segment IDs.
    """
    hstructures = load_structures(structures_dir)
    simi_matches, hfinder = find_similarities(
        hstructures, n_jobs=n_jobs,
        resampling_size=config["resampling_size"],
        num_searches=config["num_searches"],
        dist_threshold=config["dist_threshold"])
    # Preparing data structures to save all the output to disk
    simi_matches_df = pd.DataFrame(simi_matches)  # pandas edgelist
    hstructures_map = {index: id for index, id in enumerate(hstructures.keys())}
    hfinder.dump_search_model(os.path.join(out_dir, "hfinder.pkl"))
    simi_matches_df.to_csv(os.path.join(out_dir, "similarities.csv"), index=False)
    with open(os.path.join(out_dir, "pattern2id.pkl"), 'wb') as handle:
        pickle.dump(hstructures_map, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    """
    TODO
    """
    COMMANDS = ["segment", "similarities", "network"]

    parser = argparse.ArgumentParser(
        description='Main runner for the creation of Harmory.')
    parser.add_argument('cmd', type=str, choices=COMMANDS,
                        help=f"Either {', '.join(COMMANDS)}.")

    parser.add_argument('data', type=str,
                        help='Directory where JAMS files, pickles, or any dump'
                             ' will be read for further processing.')
    
    parser.add_argument('--selection', type=str,
                        help='A txt file with ChoCo IDs for song selection.')

    parser.add_argument('--config', type=list,
                        help='Configuration file with the hyperparameter set.')

    parser.add_argument('--out_dir', type=str,
                        help='Directory where all output will be saved.')
    parser.add_argument('--n_workers', action='store', type=int, default=1,
                        help='Number of workers for stats computation.')
    parser.add_argument('--compression', action='store', type=int, default=1,
                        help='Compression rate for saving the stats file.')
    parser.add_argument('--debug', action='store_true',
                        help='Whether to run in debug mode: slow and logging.')
    parser.add_argument('--log_level', action='store', default=logging.INFO,
                        help='Whether to run in debug mode: slow and logging.')

    args = parser.parse_args()
    set_logger("harmory", args.log_level)
    if args.out_dir is not None:  # sanity check and default init
        if not os.path.exists(args.out_dir):
            print(f"Creating new output directory: {args.out_dir}")
            args.out_dir = create_dir(args.out_dir)
    else:  # using the same directory of the input dataset
        args.out_dir = os.path.dirname(args.dataset)

    # Now we should be loading the config file, or config name for SEG/SIM
    config = ConfigFactory.default_config()  # FIXME XXX TODO

    if args.cmd == "segment":
        print(f"SEGMENT: Segmenting chord sequences into harmonic structures")
        # First retrieve names and build paths for the selection of tracks
        with open(args.selection, "r") as f:
            choco_ids = f.read().splitlines()
        print(f"Expected {len(choco_ids)} in {args.data}")
        jams_paths = [os.path.join(args.data, id) for id in choco_ids]
        print(f"Harmonic structure analysis started, this may take a while!")
        if not args.debug and args.n_workers > 1:
            Parallel(n_jobs=args.n_workers)(delayed(create_segmentation)\
                (jam, config=config, out_dir=args.out_dir)\
                    for jam in tqdm(jams_paths))
        else:  # this will run in debug mode, with sequential processing
            print("Running in debug mode: sequential processing in place.")
            for jam in tqdm(jams_paths):
                try:
                    create_segmentation(jam, config=config, out_dir=args.out_dir)
                except Exception as e:
                    print(f"Error at {jam} -- {e}")

    elif args.cmd == "similarities":
        print(f"SIMILARITIES: Extracting harmonic similarities in {args.data}")
        create_similarities(args.data, config=config,
            out_dir=args.out_dir, n_jobs=args.n_workers)


    else:  # trivially, args.cmd == "network"
        raise NotImplementedError()
    
    print("DONE!")


if __name__ == "__main__":
    main()
