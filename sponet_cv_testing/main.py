import sys
import logging
import numba
import json

from runmanagement import get_runfiles, run_queue

logger = logging.getLogger("cv_testing")
logger.setLevel(logging.DEBUG)

complete_file_handler = logging.FileHandler("complete_log.log")  # Handler that logs all messages to a text log
complete_file_handler.setLevel(logging.DEBUG)

test_file_handler = logging.FileHandler("testlog.log")  # Handler that just logs the important messages about the tests
test_file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

compact_formatter = logging.Formatter("%(asctime)s - %(message)s")
console_handler.setFormatter(compact_formatter)
test_file_handler.setFormatter(compact_formatter)

complete_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
complete_file_handler.setFormatter(complete_formatter)

logger.addHandler(complete_file_handler)
logger.addHandler(console_handler)
logger.addHandler(test_file_handler)


# TODO Konsolen Logging fixen. Gerade werden manche messages zweimal in die Konsole geschickt.

def setup() -> dict:
    with open("CONFIG.json", "r") as file:
        config = json.load(file)

    numba.set_num_threads(config["number_of_threads"])

    return config


def main() -> None:
    """
    Starts the runqueue form the command line.
    args:
    queue_path: str
        Path to the testfiles. Path can lead to a folder in which all testfiles are located. In this case path must end
        with "/". Path can also lead to a single json file. In this case path must end with ".json"
    work_dir_path: str
        Path to a directory where the work directories for each test are saved.
    """
    logging.info("Started main.py")
    args = sys.argv[1:]

    queue_path: str = args[0]
    work_dir_path: str = args[1]

    config = setup()

    run_files_list: list[dict] = get_runfiles(queue_path)

    run_queue(
        run_files_list,
        work_dir_path,
        config["archive_path"],
        exit_after_error=config["exit_after_error"])

    return


if __name__ == '__main__':
    main()
