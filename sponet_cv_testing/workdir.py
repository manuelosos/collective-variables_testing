import json
import os
import datetime


def create_work_dir(path: str, run_parameters: dict) -> str:
    """Creates a dir in the specified location and saves the parameter file in it.

    Parameters:
    path (str): The path in which the folder will be created.
    run_parameters (dict) : Dictionary of parameters for the test.

    Returns (str)
    Path to the created folder
    """
    run_id: str = run_parameters["run_id"]
    run_folder_path: str = f"{path}/{run_id}/"
    os.mkdir(run_folder_path)

    with open(f"{run_folder_path}parameters.json", "w") as target_file:
        json.dump(run_parameters, target_file, indent=3)

        now = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M")
        write_misc_data(path, {"run_started": now})

    return run_folder_path


def write_misc_data(path: str, data: dict) -> None:
    with open(f"{path}misc_data.txt", "a") as file:
        for key, value in data.items():
            file.write(f"{key}:{str(value)}\n")
    return
