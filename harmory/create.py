"""
Main entry point for recreating the Harmonic Memory from scratch, or extend the
memory with new harmonic progressions or patterns manually provided as inputs.

"""
import os
import pickle
import logging
import argparse

import jams
from tqdm import tqdm
from joblib import Parallel, delayed

import segmentation as seg
from config_factory import ConfigFactory
from data import create_chord_sequence, postprocess_chords
from tonalspace import TpsOffsetTimeSeries, TpsProfileTimeSeries

logger = logging.getLogger("harmory.create")


class HarmonicPrint:
    """
    The harmonic print of a progression holds relevant representations.

    TODO: imo it should contain both the time series.
    """
    def __init__(self, jams_path, sr, tpst_type="offset") -> None:
        """
        Harmonic sequence encoder from a JAMS file and symbolic sample rate.
        """
        self.sr = sr  # the symbolic sample rate
        self.id = os.path.splitext(os.path.basename(jams_path))[0]  # jaah_10
        jams_object = jams.load(jams_path, validate=False)
        # Sanity check of the requested TPS time series tpye against supported
        if tpst_type not in ["offset", "profile"]:  # supported TPS time series
            raise ValueError(f"Not a supported time series type: {tpst_type}")
        self._tpst_type = tpst_type
        # Sanity checks on the JAMS file: chord and key annotations expected
        if len(jams_object.annotations.search(namespace="chord")) == 0:
            raise ValueError("No chord annotation in JAMS file!")
        if len(jams_object.annotations.search(namespace="key")) == 0:
            raise ValueError("No key annotation in JAMS file!")
        self.metadata = jams_object.file_metadata  # just keep meta?
        # Extracting, processing, and aligning chords, keys, and times
        chords, self._keys, self._times  = \
            create_chord_sequence(jams_object, 1/sr, shift=True)
        self._chords = postprocess_chords(chords=chords)
        # Class object that will be created on demand
        self._chord_ssm = None
        self._tpsd_cache = None
        self._tps_timeseries = None

    @property
    def chord_ssm(self, normalise=True, as_distance=False, symmetric=True):
        """
        Return the TPS-SSM computed for all chord pairs in the sequence. See
        `segmentation.create_chord_ssm` for parameter specification.
        """
        if self._chord_ssm is not None:
            return self._chord_ssm
        # Creation and expansion of the TPS-SSM from the harmonic sequence
        chord_ssm, self._tpsd_cache = seg.create_chord_ssm(
            self._chords, self._keys, normalisation=normalise,
            as_distance=as_distance, symmetric=symmetric)
        self._chord_ssm = seg.expand_ssm(chord_ssm, self._times)
        logger.debug(f"TPS-SSM of shape {self._chord_ssm.shape}")
        return self._chord_ssm

    @property
    def tps_timeseries(self):
        """
        Return the TPS time series associated to this harmonic sequence. The
        type can be either offset- (sequential) or profile- (original) based.
        """
        # Return cached version, if available
        if self._tps_timeseries is not None:
            return self._tps_timeseries
        # Computing the TPS time series and updating class objects
        ts_class = TpsOffsetTimeSeries if self._tpst_type=="offset" \
            else TpsProfileTimeSeries  # parameterise time series type
        self._tps_timeseries = ts_class(
            self._chords, self._keys, self._times,
            tpsd_cache=self._tpsd_cache)
        return self._tps_timeseries


class IllegalSegmentationStateException(Exception):
    """Raised when attempting to skip segmentation steps."""
    pass


class HarmonicSegmentation:
    """
    A stateful class for harmonic segmentation, holding results incrementally.
    """
    def __init__(self, harmonic_print:HarmonicPrint) -> None:
        """
        Create a segmentation instance for harmonic structure analysis.
        """
        self.hprint = harmonic_print
        # Novelty detection data structures and parameters
        self._current_novelty = None
        self._current_l_kernel = None
        self._current_var_kernel = None
        # Peak detection algorithm and parameters
        self._pdetection_method = None
        self._pdetection_params = None
        self._current_peaks = None
        self._current_pdout = None
        # Structures resulting from the segmentation 
        self._harmonic_structures = None

    def _flush_segmentation(self):
        self._current_peaks = None
        self._current_pdout = None
        self._harmonic_structures = None

    def compute_novelty_curve(self, l:int, var:float, exclude_extremes=True):
        """
        Computes a novelty curve on the harmonic SSM through correlation of
        a Gaussian kernel, of size l*sr and variance `var`, along the main
        diagonal. This value is recomputed if using different parameters.

        Parameters
        ----------
        l : int
            The size, in seconds, of the Gaussian kernel to use.
        var : float
            Variance of the Gaussian kernel to consider.

        Returns
        -------
        novelty_curve : np.ndarray
            The novelty curve computed on the SM using the given parameters.

        """
        if self._current_novelty is not None and \
            self._current_l_kernel == l and self._current_var_kernel == var:
            return self._current_novelty  # use cached version
        # Flush current segmentation results, as not consistent anymore
        self._flush_segmentation()
        # Cache any changed parameter for novelty detection
        self._current_l_kernel, self._current_var_kernel = l, var

        sl = l * self.hprint.sr  # from seconds to samples
        hssm = self.hprint.chord_ssm  # triggers computation 1st time
        self._current_novelty = seg.compute_novelty_sm(
            hssm, L=sl, var=var, exclude=exclude_extremes)

        return self._current_novelty

    def detect_peaks(self, pdetection_method:str, **pdetection_args):
        """
        Performs peak detection on the previously computed novelty curve, using
        the chosen algorithm for peak picking and method-specific parameters.
        Detected peaks are then cached for future calls.

        Parameters
        ----------
        pdetection_method : str
            The name of a supported peak detection algorithm to use; one of 
            segmentation.PDETECTION_MAP`.
        pdetection_args : dict
            A mapping providing method-specific arguments for peak picking. If
            none is provided, the default ones will be used.

        Returns
        -------
        peaks : np.ndarray
            Temporal locations (in samples!) of the detected peaks.
        extra_output : tuple
            Any additional output returned by the peak detection algorithm.

        Raises
        ------
        IllegalSegmentationStateException
            If this step is performed before computing the novelty curve.
        ValueError
            If the requested peak picking method is not supported, yet.
        
        """
        if self._current_peaks is not None and \
            self._pdetection_method == pdetection_method and \
                self._pdetection_params == pdetection_args:
                return self._current_peaks  # use cached version

        if self._current_novelty is None:  # illegal state change
            raise IllegalSegmentationStateException("Compute novelty first!")
        if pdetection_method not in seg.PDETECTION_MAP:
            raise ValueError(f"Not supported algorithm: {pdetection_method}")
        # Flush current segmentation results, and update parameters
        self._flush_segmentation()
        self._pdetection_method, self._pdetection_params = \
            pdetection_method, pdetection_args
        # Perform peak detection with the chosen method and params
        pdetection_fn = seg.PDETECTION_MAP.get(pdetection_method)
        pd_output = pdetection_fn(self._current_novelty, **pdetection_args)
        self._current_peaks, self._current_pdout = pd_output[0], pd_output[1:]

        return self._current_peaks, self._current_pdout

    def segment_harmonic_print(self):
        """
        Returns the segmented harmonic structure following peak detection.

        Returns
        -------
        harmonic_structure : List(tonalpitchspace.TpsTimeSeries)
            The list of harmonic structures as TPS time series.

        Raises
        ------
        IllegalSegmentationStateException
            If attempting to segment before peak detection.

        """
        if self._harmonic_structures is not None:
            return self._harmonic_structures  # use cached version
        if self._current_peaks is None:  # illegal state change
            raise IllegalSegmentationStateException("Detect peaks first!")
        # Splitting the original time series of the harmonic print
        self._harmonic_structures = self.hprint.tps_timeseries.\
            segment_from_times(self._current_peaks)
        return self._harmonic_structures

    def dump_harmonic_segments(self, out_dir):
        """
        Saves the detected harmonic structures in a pickle file, using the
        identifier of the former chord sequences (e.g. isophonics_0.pickle).
        A list indexing the TPSTimeSeries of the harmonic structures is used.
        """
        if self._harmonic_structures is None:  # illegal state change
            raise IllegalSegmentationStateException("Segmentation required!")

        fpath = os.path.join(out_dir, f"{self.hprint.id}.pickle")
        logger.debug(f"Saving {len(self._harmonic_structures)} in {fpath}")
        with open(fpath, 'wb') as handle:
            pickle.dump(self._harmonic_structures, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    
    def run(self, l_kernel, var_kernel, pdetection_method, **pdetection_args):
        """
        Performs all the steps above for harmonic structure analysis.
        """
        self.compute_novelty_curve(l_kernel, var_kernel)
        self.detect_peaks(pdetection_method, **pdetection_args)
        return self.segment_harmonic_print()



def create_segmentation(jams_path, config, out_dir):

    hprint = HarmonicPrint(jams_path,
        sr=config["sr"], tpst_type=config["tpst_type"])
    hprint.chord_ssm  # this will init internal structures
    hprint._tps_timeseries  # this will init internal structures
    segmenter = HarmonicSegmentation(hprint)

    segmenter.run(
        l_kernel=config["l_kernel"],
        var_kernel=config["var_kernel"],
        pdetection_method=config["pdetection_method"],
        **config["pdetection_params"])

    segmenter.dump_harmonic_segments(out_dir)



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

    args = parser.parse_args()
    if args.out_dir is not None:  # sanity check and default init
        if not os.path.exists(args.out_dir):
            raise ValueError(f"Directory {args.out_dir} does not exist!")
    else:  # using the same directory of the input dataset
        args.out_dir = os.path.dirname(args.dataset)

    if args.cmd == "segment":
        print(f"SEGMENT: Segmenting chord sequences into harmonic structures")
        # First retrieve names and build paths for the selection of tracks
        with open(args.selection, "r") as f:
            choco_ids = f.read().splitlines()
        print(f"Expected {len(choco_ids)} in {args.data}")
        jams_paths = [os.path.join(args.data, id) for id in choco_ids]
        # Now we should be loading the config file, or config name for SEG
        config = ConfigFactory.default_config()  # FIXME XXX TODO
        print(f"Harmonic structure analysis started, this may take a while!")
        Parallel(n_jobs=args.n_workers)(delayed(create_segmentation)\
            (jam, config=config, out_dir=args.out_dir)\
                for jam in tqdm(jams_paths))


    elif args.cmd == "similarities":
        print(f"SIMILARITIES: Extracting harmonic similarities in {args.data}")
        # First read/load the joblib file containing the JAMS stats
        raise NotImplementedError()
    else:  # trivially, args.cmd == "network"
        raise NotImplementedError()


if __name__ == "__main__":
    main()
