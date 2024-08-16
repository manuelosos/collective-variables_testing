import logging
import shutil
import json
import pandas as pd
import os
import sys
import networkx as nx
import datetime
import numpy as np
from dataclasses import dataclass
import sponet_cv_testing.resultmanagement as rm

# global variables that should be changed if necessary
if __name__ == "__main__":
    data_path: str = "../data/"
    #data_path: str = "../tests/test_data/"
    logger = logging.getLogger("cv_testing.datamanagement")
else:
    data_path: str = "/home/manuel/Documents/Studium/praktikum/code/sponet_cv_testing/data/"
    #data_path = "/home/manuel/Documents/Studium/praktikum/code/sponet_cv_testing/tests/test_data/"
results_csv_path: str = "results/results_table.csv"


def archive_run_result(source: str) -> None:

    with open(f"{source}parameters.json", "r") as target_file:
        parameters: dict = json.load(target_file)

    run_id: str = str(parameters["run_id"])

    file_list: list[str] = os.listdir(source)
    if "run_finished.txt" not in file_list:
        print(f"Run {run_id} not finished! Run will not be archived.")
        return

    finished = True


    dynamic_parameters: dict = parameters["dynamic"]
    dynamic_model: str = dynamic_parameters["model"]
    dynamic_rates: tuple = _translate_run_rates(dynamic_parameters["rates"])

    network_parameters: dict = parameters["network"]
    network_id = network_parameters.get("network_id", "")
    network_model = network_parameters["model"]

    if network_parameters["generate_new"]:
        num_nodes = network_parameters["num_nodes"]
    else:
        network = rm.open_network(f"{source}networks", "network")
        num_nodes = network.number_of_nodes()

    sampling_parameters: dict = parameters["simulation"]["sampling"]
    lag_time: float = sampling_parameters["lag_time"]
    num_time_steps: int = sampling_parameters.get("num_timesteps",1)
    num_anchor_points: int = sampling_parameters["num_anchor_points"]
    num_samples_per_anchor: int = sampling_parameters["num_samples_per_anchor"]
    num_coordinates: int = parameters["simulation"]["num_coordinates"]


    dimension_estimate = rm.get_dimension_estimate(source)[-1]

    results = read_data_csv()

    """
    overwrite: bool = False
    if run_id in results.index:
        # If already archived run is unfinished and new run is finished,
        # the new finished run will overwrite the archived one.
        if not results.loc[run_id]["finished"] and finished:
            overwrite = True
            logger.info(f"Unfinished run with id {run_id} will be overwritten with new finished run.")
        else:
            #raise FileExistsError("The run id is not unique")
            logger.info(f"Run is not unique and not successful run {run_id} will be skipped.")
            return
    """

    remarks = read_logs(source)

    remarks += " " + parameters.get("remark", "")

    new_result: list = [
        dynamic_model, *dynamic_rates, network_id, network_model, num_nodes, lag_time, num_time_steps,
        num_anchor_points, num_samples_per_anchor, num_coordinates, dimension_estimate, finished, remarks]


    if run_id in results.index:
        shutil.rmtree(f"{data_path}results/{run_id}/")

    results.loc[run_id] = new_result

    logger.debug(f"Starting moving files from {source} to {data_path}results/")

    shutil.move(source, f"{data_path}results/")
    logger.info(f"Finished moving files from {source} to {data_path}results/")
    save_csv(results)
    logger.info(f"archived run {run_id} in csv.")

    return

def read_data_csv(path: str=f"{data_path}{results_csv_path}") -> pd.DataFrame:
    """
    Parameters
    ----------
    path : (str)
        path to the results_csv.
    Returns
    -------
        DataFrame containing the relevant values of every run.

    """
    data_csv = pd.read_csv(
        path,
        index_col=0,
        dtype={
            "run_id": str,
            "dynamic_model": str,
            "r_ab": float,
            "r_ba": float,
            "rt_ab": float,
            "rt_ba": float,
            "network_id": str,
            "network_model": str,
            "num_nodes": int,
            "lag_time": float,
            "num_time_steps": int,
            "num_anchor_points": int,
            "num_samples_per_anchor": int,
            "cv_dim": int,
            "dim_estimate": float,
            "finished": bool,
            "remarks": str}
    )
    return data_csv


def save_csv(df: pd.DataFrame) -> None:

    df.to_csv(f"{data_path}{results_csv_path}")
    return


def unique_network_id(network_id: str) -> bool:
    network_list = os.listdir(f"{data_path}networks")
    if network_id in network_list:
        return False
    else:
        return True


def generate_network_id(parameters) -> str:

    num_nodes = parameters["num_nodes"]
    model = parameters["model"]
    if model == "albert-barabasi":
        num_attachments = parameters["num_attachments"]
        network_id = f"albert-barabasi_{num_nodes}n_{num_attachments}a"
    else:
        raise ValueError(f"Unknown model {model}")

    counter = 0
    tmp = network_id
    while not unique_network_id(tmp):
        counter += 1
        tmp = f"{network_id}_{counter}"
    network_id = tmp
    return network_id


def open_network(path: str, network_id: str) -> nx.Graph:
    return nx.read_graphml(f"{path}{network_id}")


def read_logs(path: str) -> str:

    remarks: list[str] = []
    with open(f"{path}runlog.log", "r") as logfile:
        logs = logfile.readlines()

    for line in logs:
        if "#slb" in line:
            remarks.append("#slb")
            continue

    return " ".join(remarks)


def change_run_id(run_id: str, new_run_id: str) -> None:

    raise NotImplementedError
    # TODO zum laufen bringen gerade zerschießt es die Daten

    with open(f"{data_path}results/{run_id}/parameters.json", "r") as runfile:
        parameters = json.load(runfile)

    parameters["run_id"] = new_run_id

    result_df = read_data_csv()


    result_df.at["run_id", run_id] = new_run_id

    with open(f"{data_path}results/{run_id}/parameters.json", "w") as runfile:
        json.dump(parameters, runfile, indent=3)
    save_csv(result_df)

    logger.info(f"#ridch {run_id} -> {new_run_id}")
    return


def _translate_run_rates(rates: dict) -> tuple[float, float, float, float]:
    """Returns the rates of the run in a list. Only accepts Type 1 Parameters of CNVM."""
    r: np.ndarray = np.array(rates["r"])
    r_tilde: np.ndarray = np.array(rates["r_tilde"])
    return r[0, 1], r[1, 0], r_tilde[0, 1], r_tilde[1, 0]





def archive_dir(path: str) -> None:

    dir_list: list = os.listdir(path)
    for file in dir_list:
        archive_run_result(f"{path}{file}/")
    return


def unique_run_id(run_id: str) -> bool:
    """Checks if a run_id is unique in the archive."""
    df = read_data_csv()
    return run_id not in df.index


def generate_unique_run_id(n: int = 1, name: str = "") -> list[str]:

    timestamp: str = datetime.datetime.now().strftime("%y-%m-%d")

    if not name == "":
        name = f"_{name}"

    results = read_data_csv()

    counter = 0
    while True:
        if f"{timestamp}{name}_{counter}" not in results.index:
            break
        counter += 1

    run_ids: list[str] = []
    for i in range(n):
        run_ids.append(f"{timestamp}{name}_{counter+i}")

    return run_ids


class RunResult:
    run_id: str
    dynamic_model: str
    r_ab: float
    r_ba: float
    rt_ab: float
    rt_ba: float
    network_id: str | None
    network_model: str
    num_nodes: int
    lag_time: float
    num_anchor_points: int
    num_samples_per_anchor: int
    cv_dim: int
    dim_estimate: float
    finished: bool


if __name__ == "__main__":



    archive_dir("../tests/tmp_results/")


