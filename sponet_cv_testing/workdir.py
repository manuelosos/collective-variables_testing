import json
import os
import datetime as dt
import numpy as np


def create_work_dir(path: str, run_parameters: dict) -> str:
    """Creates a directory in the specified path and creates some metadata files.

    Parameters:
    path (str) : The path in which the folder will be created.
    run_parameters (dict) : Dictionary of parameters for the run.

    Returns (str)
    Path to the created folder
    """
    run_id: str = run_parameters["run_id"]
    run_folder_path: str = f"{path}/{run_id}/"
    os.mkdir(run_folder_path)
    os.mkdir(f"{run_folder_path}misc_data/")

    remarks = run_parameters.get("remarks", "")

    with open(f"{run_folder_path}misc_data/remarks.txt", "w") as remarks_file:
        remarks_file.write(remarks)

    with open(f"{run_folder_path}parameters.json", "w") as target_file:
        json.dump(run_parameters, target_file, indent=3)

    now = dt.datetime.now().strftime("%d.%m.%Y %H:%M")
    write_metadata(run_folder_path, {"run_started": now})

    return run_folder_path


def write_metadata(run_folder_path: str, data: dict) -> None:
    with open(f"{run_folder_path}/misc_data/meta_data.txt", "a") as file:
        for key, value in data.items():
            file.write(f"{key}:{str(value)}\n")
    return


def save_compute_times(path: str, times: np.ndarray, header="") -> None:
    times = np.array([str(dt.timedelta(seconds=t)) for t in times])
    np.savetxt(path, times, fmt="%s", header=header)
    return

