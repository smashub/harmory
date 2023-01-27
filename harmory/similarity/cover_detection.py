"""
Script to detect cover songs in a dataset of songs.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm


EXPERIMENTS = [('tpsd', 'offset'),
               ('tpsd', 'profile'),
               ('dtw', 'offset'),
               ('dtw', 'profile')]

def main():
    pass

if __name__ == '__main__':
    main()
