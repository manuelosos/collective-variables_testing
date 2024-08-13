import json
import os
import datetime
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

    with open(f"{run_folder_path}parameters.json", "w") as target_file:
        json.dump(run_parameters, target_file, indent=3)

    now = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    write_metadata(run_folder_path, {"run_started": now})

    return run_folder_path


def write_metadata(run_folder_path: str, data: dict) -> None:
    with open(f"{run_folder_path}/misc_data/meta_data.txt", "a") as file:
        for key, value in data.items():
            file.write(f"{key}:{str(value)}\n")
    return


def save_array(path: str, data: np.ndarray) -> None:
    """
    Saves an array in the specified path.
    Parameters
    ----------
    path : str
        Path to which the data is saved. This includes the name of the file.
        If any parent directory does not exist, it will be created.
        The file extension is automatically appended.
    data : np.ndarray
        Array to be saved.

    Returns None
    """



    return
