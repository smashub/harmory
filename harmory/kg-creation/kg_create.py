"""
Main script for creating the KG from the file produces in Harmory.
"""

import argparse
import logging
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import tqdm

from collections import defaultdict