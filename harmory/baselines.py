"""
Run segmentation baselines on a given datasets, using a configuration grid.
"""
import os
import logging
import argparse
import itertools

from tqdm import tqdm

import harmseg as hseg
from config import ConfigFactory
from utils import create_dir, set_logger

logger = logging.getLogger("harmory.baselines")

N_REGIONS_GRID = [10, 11, 12, 13, 14]
MPROFILE_GRID = [3]


BASELINE_PARAM_GRID = {
    "random_split": {
        "n_regions": N_REGIONS_GRID,
    },
    "uniform_split": {
        "n_regions": N_REGIONS_GRID,
    },
    "quasirandom_split": {
        "n_regions": N_REGIONS_GRID,
    },
    "fluss_segmentation": {
        "n_regions": N_REGIONS_GRID,
        "m": MPROFILE_GRID,
    }
}


def generate_grid_instances(param_grid: dict):
    keys, values = zip(*param_grid.items())
    return[dict(zip(keys, v)) for v in itertools.product(*values)] 

def paramset_to_str(param_set: dict):
    return "__".join([f"{k}%{v}" for k,v in param_set.items()])


def segmentation_baselines(jams_path, baseline_grid, config, out_dir):

    hprint = hseg.HarmonicPrint(jams_path,
        sr=config["sr"], tpst_type=config["tpst_type"])
    hprint.run()  # creates SSM and TPS time series

    for baseline in baseline_grid.keys():  # assume all are legit
        logger.info(f"Processing baseline {baseline} on {jams_path}")
        segmenter_fn = hseg.SEGMENTATION_BASELINE_FNS.get(baseline)
        segmenter = hseg.TimeSeriesHarmonicSegmentation(hprint, segmenter_fn)
        baseline_outdir = os.path.join(out_dir, baseline)
        # From a baseline GRID to all possible parameter sets drawn from it 
        segmenter_grid = generate_grid_instances(baseline_grid[baseline])
        for param_set in segmenter_grid:  # perform grid search and dump
            segmenter.run(**param_set)
            segmenter.dump_harmonic_segments(
                os.path.join(baseline_outdir, paramset_to_str(param_set)))


def create_baseline_setup(baseline_grid, out_dir):
    """Needed to create baseline-specific folders where data will be saved."""
    if not os.path.isdir(os.path.dirname(os.path.dirname(out_dir))):
        raise ValueError("Upper directory %s does not exist!" % out_dir)

    exp_dir = create_dir(out_dir)  # root dir for the experiment
    for baseline_name, baseline_grid in baseline_grid.items():
        baseline_outdir = create_dir(os.path.join(exp_dir, baseline_name))
        baseline_param_sets = generate_grid_instances(baseline_grid)
        logger.info(f"Found {len(baseline_param_sets)} parameter sets for "
                    f"baseline {baseline_name}")
        for baseline_parameter_set in baseline_param_sets:
            base_pset_dir = create_dir(os.path.join(  # EXP > BASELINE > PARSET
                baseline_outdir, paramset_to_str(baseline_parameter_set)))
            logger.info(f"Creating baseline-pset dir at {base_pset_dir}")



def main():

    parser = argparse.ArgumentParser(
        description='Main runner for the harmonic segmentation baselines.')

    parser.add_argument('data', type=str,
                        help='Directory where JAMS files, pickles, or any dump'
                             ' will be read for further processing.')
    
    parser.add_argument('--selection', type=str,
                        help='A txt file with ChoCo IDs for song selection.')

    parser.add_argument('--grid', type=list,
                        help='Configuration file with the hyperparameter set.')
    parser.add_argument('--baselines', type=str, nargs="*",
                        help='Name of the baseline for this experiment.')

    parser.add_argument('--out_dir', type=str,
                        help='Directory where all output will be saved.')
    parser.add_argument('--n_workers', action='store', type=int, default=1,
                        help='Number of workers for stats computation.')
    parser.add_argument('--debug', action='store_true',
                        help='Whether to run in debug mode: slow and logging.')
    parser.add_argument('--log_level', action='store', default=logging.INFO,
                        help='Whether to run in debug mode: slow and logging.')

    args = parser.parse_args()
    set_logger("harmory", args.log_level)
    # Now we should be loading the config file, or config name for SEG/SIM
    baseline_grid = BASELINE_PARAM_GRID if args.baselines is None else \
        {n: g for n, g in BASELINE_PARAM_GRID.items() if n in args.baselines}
    config = ConfigFactory.default_config()  # FIXME XXX TODO
    print(f"Grid search parameters: {baseline_grid}")
    create_baseline_setup(baseline_grid, args.out_dir)

    print(f"SEGMENTATION baseline runner")
    # First retrieve names and build paths for the selection of tracks
    with open(args.selection, "r") as f:
        choco_ids = f.read().splitlines()
    print(f"Expected {len(choco_ids)} in {args.data}")
    jams_paths = [os.path.join(args.data, id) for id in choco_ids]
    print(f"Grid search is starting, this may take a while!")

    for jams_path in tqdm(jams_paths):
        try:
            segmentation_baselines(jams_path, baseline_grid, config, args.out_dir)
        except Exception as e:
                    logger.error(f"Error at {jams_path} -- {e}")


    # if not args.debug and args.n_workers > 1:
    #     Parallel(n_jobs=args.n_workers)(delayed(create_segmentation)\
    #         (jam, config=config, out_dir=args.out_dir)\
    #             for jam in tqdm(jams_paths))
    # else:  # this will run in debug mode, with sequential processing
    #     print("Running in debug mode: sequential processing in place.")
    #     for jam in tqdm(jams_paths):
    #         try:
    #             create_segmentation(jam, config=config, out_dir=args.out_dir)
    #         except Exception as e:
    #             print(f"Error at {jam} -- {e}")

    print("DONE!")


if __name__ == "__main__":
    main()
