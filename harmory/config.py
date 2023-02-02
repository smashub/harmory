"""
Temporary Factory of config files; will be extended and refactored as WIP.

"""
import constants

# XXX Ugly for now, but this is supposed to hold the best configuration
default_config = {
    "sr": constants.SR,
    "l_kernel": constants.L_KERNEL,
    "var_kernel": constants.VAR_KERNEL,
    "tpst_type": "profile",
    "pdetection_method": "msaf",
    "pdetection_params": dict(median_len=24, sigma=2),
    "resampling_size": constants.RESAMPLING_SIZE,
    "num_searches": constants.N_SEARCHES,
    "dist_threshold": constants.DIST_THRESHOLD,
}


class ConfigFactory:

    @staticmethod
    def default_config():
        return default_config

    @staticmethod
    def create_config():
        raise NotImplementedError()

