"""
Temporary Factory of config files; will be extended and refactored as WIP.

"""
import constants

# XXX Ugly for now, but this is supposed to hold the best configuration
default_config = {
    "sr": constants.SR,
    "l_kernel": constants.L_KERNEL,
    "var_kernel": constants.VAR_KERNEL,
    "tpst_type": "offset",
    "pdetection_method": "msaf",
    "pdetection_params": dict(median_len=24, sigma=2)
}


class ConfigFactory:

    @staticmethod
    def default_config():
        return default_config

    @staticmethod
    def create_config():
        raise NotImplementedError()

