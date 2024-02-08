import logging
import shutil
import json
import pandas as pd
import os
import networkx as nx
import datetime

# global variables that should be changed if necessary
data_path: str = "data/"
results_csv_path: str = "results/results_table.csv"

logger = logging.getLogger("sponet_cv_testing.datamanagement")
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


def archive_network(source: str, network_parameters: dict) -> tuple:

    network_id = str(network_parameters["network_id"])

    if network_parameters["generate_new"]:
        file_list = os.listdir(f"{data_path}networks")
        if network_id in file_list:
            logger.error(f"Network {network_id} is already saved or has non unique id.")
            raise ValueError(f"Network {network_id} is already saved or has non unique id.")

        shutil.move(f"{source}/{network_id}", f"{data_path}networks/")
        name = network_parameters["name"]
        num_nodes = network_parameters["num_nodes"]

    else:
        network: nx.Graph = nx.read_graphml(f"{data_path}networks/{network_id}")
        name = network.name
        num_nodes = network.number_of_nodes()

    return network_id, name, num_nodes


def archive_run_result(source: str) -> None:

    with open(f"{source}/parameters.json", "r") as target_file:
        parameters: dict = json.load(target_file)

    run_id: str = str(parameters["run_id"])

    dynamic_parameters: dict = parameters["dynamic"]
    dynamic_name: str = dynamic_parameters["name"]
    num_states: int = dynamic_parameters["num_states"]

    network_parameters: dict = parameters["network"]
    network_id, network_type, num_nodes = archive_network(source, network_parameters)

    sampling_parameters: dict = parameters["simulation"]["sampling"]
    lag_time: float = sampling_parameters["lag_time"]
    num_anchor_points: int = sampling_parameters["num_anchor_points"]
    num_samples_per_anchor: int = sampling_parameters["num_samples_per_anchor"]

    cv_dimension: int = parameters["simulation"]["num_coordinates"]

    results = pd.read_csv(
        f"{data_path}{results_csv_path}",
        index_col=0,
        dtype={"run_id": str}
    )

    if run_id in results.index:

        logger.error(f"Run {run_id} has no unique id")
        raise FileExistsError("The run id is not unique")

    file_list: list[str] = os.listdir(source)
    if "run_finished.txt" not in file_list:
        logger.error(f"Run {run_id} has not been finished. Archiving failed")
        raise ValueError(f"Run {run_id} is not finished")

    new_result: list = [dynamic_name, num_states, network_id, network_type,
                        num_nodes, lag_time, num_anchor_points, num_samples_per_anchor, cv_dimension]

    results.loc[run_id] = new_result
    results.to_csv(f"{data_path}{results_csv_path}")
    logger.info(f"archived run {run_id} in csv.")

    logger.debug(f"Starting moving files from {source} to {data_path}results/")
    shutil.move(source, f"{data_path}results/")
    logger.info(f"Finished moving files from {source} to {data_path}results/")

    return


def generate_new_unique_run_id(n: int = 1, name: str = "") -> list[str]:

    timestamp: str = datetime.datetime.now().strftime("%y-%m-%d")

    if not name == "":
        name = f"_{name}"

    results = pd.read_csv(
        f"{data_path}{results_csv_path}",
        index_col=0,
        dtype={"run_id": str}
    )

    counter = 0
    while True:
        if f"{timestamp}{name}_{counter}" not in results.index:
            break
        counter += 1

    run_ids: list[str] = []
    for i in range(n):
        run_ids.append(f"{timestamp}{name}_{counter+i}")

    return run_ids


if __name__ == "__main__":
    archive_run_result("/home/manuel/Documents/Studium/praktikum/code/sponet_cv_testing/sponet_cv_testing/tmp_results/1")
