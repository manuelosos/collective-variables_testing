import sys
import logging
import numba
import json
import os
import argparse
from sponet_cv_testing.runmanagement import run_queue

logger = logging.getLogger("testpipeline")
logger.setLevel(logging.DEBUG)

# Handler that logs ALL messages to a text log
complete_file_handler = logging.FileHandler("complete_log.log")
complete_file_handler.setLevel(logging.DEBUG)

# Handler that sends progress-relevant messages to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

compact_formatter = logging.Formatter("%(name)s - %(asctime)s - %(message)s")
console_handler.setFormatter(compact_formatter)

complete_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
complete_file_handler.setFormatter(complete_formatter)

logger.addHandler(complete_file_handler)
logger.addHandler(console_handler)


# Setting up cli arguments
parser = argparse.ArgumentParser()
parser.add_argument("-run", "--runfile_path",
                    help="Path to the directory where the runfiles are located.")

parser.add_argument("-res", "--result_path",
                    help="Path to the directory where the results will be saved.")

parser.add_argument("-net", "--network_path",
                    help="Path to the directory where the networks are located.")

parser.add_argument("-num_t", "--num_threads",
                    type=int,
                    help="Max number of threads that will be available for "
                         "computation.")

parser.add_argument("--delete_samples",
                    action="store_true",
                    help="If set, the samples from the network_dynamics will not be saved. "
                         "The dynamics samples utilize the most space on disc.")

parser.add_argument("--delete_runfile",
                    action="store_true",
                    help="If specified, runfiles will be deleted from the runfile folder after successful execution")

parser.add_argument("--error_exit",
                    action="store_true",
                    help="Path to the dir where the runfiles are located.")

parser.add_argument("--overwrite_results",
                    action="store_true",
                    help="If set to true, existing results in result_path will be overwritten if the run_id is equal.")

parser.add_argument("--device",
                    help="Device name that will be saved in the metadata.")


def main() -> None:
    """
    Starts the runqueue from the command line.
    args:
    queue_path: str
        Path to the runfiles. Path can lead to a folder in which all runfiles are located. In this case path must end
        with "/". Path can also lead to a single json file. In this case path must end with ".json"
    result_dir_path: str
        Path to a directory where the result-directories for each run are saved.
    network_dir_path: str (optional)
     Path to a directory where the networks are saved.
    """
    logging.info("Started main.py")

    with open("CONFIG.json", "r") as file:
        config = json.load(file)

    args = parser.parse_args()

    if args.runfile_path: runfile_path = args.runfile_path
    else: runfile_path = config.get("runfile_path", None)
    if not runfile_path: raise ValueError("No runfile path specified.")

    if args.result_path: result_path = args.result_path
    else: result_path = config.get("result_path", None)
    if not result_path: raise ValueError("No result path specified.")

    if args.network_path: network_path = args.network_path
    else: network_path = config.get("network_path", None)
    if network_path is None: raise ValueError("No network path specified.")

    if args.num_threads: num_threads = args.num_threads
    else: num_threads = config.get("num_threads", None)
    if num_threads:
        numba.set_num_threads(num_threads)

    if args.delete_samples: delete_samples = True
    else: delete_samples = config.get("delete_samples", False)

    if args.delete_runfile: delete_runfile = True
    else: delete_runfile = config.get("delete_runfile", False)

    if args.error_exit: error_exit = True
    else: error_exit = config.get("error_exit", False)

    if args.overwrite_results: overwrite_results = True
    else: overwrite_results = config.get("overwrite_results", False)

    if args.device: device = args.device
    else: device = config.get("device", "unknown")

    misc_data: dict = {"device": device,
                       "number of numba threads": num_threads,
                       "cpu count": os.cpu_count()}


    run_queue(
        runfile_path,
        result_path,
        network_path,
        delete_samples=delete_samples,
        delete_runfiles=delete_runfile,
        exit_after_error=error_exit,
        misc_data=misc_data,
        overwrite_results=overwrite_results)

    return


if __name__ == '__main__':
    main()

sys.exit(0)
