import sys
import logging
import json
import os
import argparse
from sponet_cv_testing.runmanagement import run_queue

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("-run", "--runfile_path",
                    help="Path to the directory where the runfiles are located.")
parser.add_argument("-res", "--result_path",
                    help="Path to the directory where the results will be saved.")