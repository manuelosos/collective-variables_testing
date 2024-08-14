import sys
import logging
import numba
import json
import os


from sponet_cv_testing.runmanagement import get_runfiles, run_queue

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


def setup() -> tuple[dict, dict]:
    with open("CONFIG.json", "r") as file:
        config = json.load(file)

    numba.set_num_threads(config["number_of_threads"])
    cpu_count = os.cpu_count()

    misc_data: dict = {"device": config["device"],
                       "number of numba threads": config["number_of_threads"],
                       "cpu count": cpu_count}

    return config, misc_data


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

    config, misc_data = setup()

    args = sys.argv[1:]
    queue_path: str = args[0]
    work_dir_path: str = args[1]

    if len(args) > 2:
        network_dir_path = args[2]
    else:
        network_dir_path = config.get("network_dir_path", None)

    run_files_list: list[dict] = get_runfiles(queue_path)

    run_queue(
        run_files_list,
        work_dir_path,
        network_dir_path,
        exit_after_error=config["exit_after_error"],
        misc_data=misc_data)

    return


if __name__ == '__main__':
    main()

sys.exit(0)
