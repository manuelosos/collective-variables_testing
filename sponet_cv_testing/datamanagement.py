import logging
import shutil
import json
import pandas as pd
import os
import networkx as nx
import datetime
import numpy as np
import csv
import re
from dataclasses import dataclass

# global variables that should be changed if necessary
if __name__ == "__main__":
    data_path: str = "../data/"
    logger = logging.getLogger("cv_testing.datamanagement")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(f"{data_path}data_log.log")
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    compact_formatter = logging.Formatter("%(asctime)s - %(message)s")
    console_handler.setFormatter(compact_formatter)

    complete_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(complete_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
else:
    data_path: str = "/home/manuel/Documents/Studium/praktikum/code/sponet_cv_testing/data/"

results_csv_path: str = "results/results_table.csv"









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


def read_misc_data(source: str) -> dict:
    with open(f"{source}misc_data.txt", "r") as file:
        reader = csv.reader(file, delimiter=":")
        data = {}
        for row in reader:
            data.update({row[0]: row[1]})
    return data


def write_misc_data(path: str, data: dict) -> None:
    with open(f"{path}misc_data.txt", "a") as file:
        for key, value in data.items():
            file.write(f"{key}:{str(value)}\n")
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


def archive_network(source: str, parameters: dict) -> str:

    if "network_id" in parameters.keys():
        network_id = str(parameters["network_id"])
        if not unique_network_id(network_id):
            network_id = generate_network_id(parameters)
            logger.warning(f"Network_id was not unique. Assigned new one: {network_id}")
    else:
        network_id = generate_network_id(parameters)

    shutil.copy(f"{source}network", f"{data_path}networks/{network_id}")
    return network_id


def open_network(path: str, network_id: str) -> nx.Graph:
    return nx.read_graphml(f"{path}{network_id}")


def save_network(network: nx.Graph, save_path: str, filename: str) -> None:
    nx.write_graphml(network, f"{save_path}{filename}")
    return


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


def archive_run_result(source: str) -> None:

    with open(f"{source}parameters.json", "r") as target_file:
        parameters: dict = json.load(target_file)

    run_id: str = str(parameters["run_id"])

    dynamic_parameters: dict = parameters["dynamic"]
    dynamic_model: str = dynamic_parameters["model"]
    dynamic_rates: tuple = _translate_run_rates(dynamic_parameters["rates"])

    network_parameters: dict = parameters["network"]
    if network_parameters["generate_new"]:
        if "archive" in network_parameters.keys() and network_parameters["archive"]:
            network_id = archive_network(source, network_parameters)
        else:
            network_id = np.nan  # Network id is given when the network is reused for another test.
        network_model = network_parameters["model"]
        num_nodes = network_parameters["num_nodes"]
    else:
        network_id = network_parameters["network_id"]
        network = open_network(f"{data_path}networks/", network_id)
        num_nodes = network.number_of_nodes()
        network_model = network.name.split["_"][0]

    sampling_parameters: dict = parameters["simulation"]["sampling"]
    lag_time: float = sampling_parameters["lag_time"]
    num_anchor_points: int = sampling_parameters["num_anchor_points"]
    num_samples_per_anchor: int = sampling_parameters["num_samples_per_anchor"]
    num_coordinates: int = parameters["simulation"]["num_coordinates"]


    file_list: list[str] = os.listdir(source)
    if "run_finished.txt" in file_list:
        finished = True
        misc_data = read_misc_data(source)
        dimension_estimate: float = float(misc_data["dimension_estimate"])
    else:
        finished = False
        dimension_estimate = np.nan

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
            logger.info(f"Run is not unique and not sucesful run {run_id} will be skipped.")
            return
    """

    remarks = read_logs(source)

    new_result: list = [
        dynamic_model, *dynamic_rates, network_id, network_model, num_nodes, lag_time, num_anchor_points,
        num_samples_per_anchor, num_coordinates, dimension_estimate, finished, remarks]

    if run_id in results.index:
        shutil.rmtree(f"{data_path}results/{run_id}/")

    results.loc[run_id] = new_result


    logger.debug(f"Starting moving files from {source} to {data_path}results/")

    shutil.move(source, f"{data_path}results/")
    logger.info(f"Finished moving files from {source} to {data_path}results/")
    save_csv(results)
    logger.info(f"archived run {run_id} in csv.")

    return


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

    #df = read_data_csv()
    #df = df.replace("[]", "")

    #print(df["remarks"] == "[]")

    #df.to_csv("test")

    #df = read_data_csv()

    #print(df.columns)
    #df[df["num_anchor_points"] == 2000].to_csv("new_data.csv")



    #print(df["remarks"])
# TODO Erfolgreiche Reruns zulassen
