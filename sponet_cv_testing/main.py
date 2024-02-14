import sys
import logging

from runmanagement import get_runfiles, run_queue

logger = logging.getLogger("sponet_cv_testing")
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


def main() -> None:

    args = sys.argv[1:]

    queue_path: str = args[0]
    work_dir_path: str = args[1]

    if len(args) == 3:
        archive_path: str = args[2]
    else:
        archive_path: str = "data/"

    run_files_list: list[dict] = get_runfiles(queue_path)

    run_queue(run_files_list, work_dir_path, archive_path)

    return


if __name__ == '__main__':
    main()
