import os
import json
import time
import logging
import networkx as nx
from sponet import network_generator as ng
from computation.compute import compute_run

logger = logging.getLogger("sponet_cv_testing.runmanagement")


def get_runfiles(path: str) -> list[dict]:
    """
    Reads the json files in a folder and returns them as dictionaries.
    Lists will not be converted to numpy arrays.

    Parameters
    path (str) : Path to the folder containing the json files.

    Returns  (list[dict])
    list of dictionaries containing the parsed json files
    """
    runfiles = os.listdir(path)

    run_parameters = []
    for runfile in runfiles:
        if runfile.endswith(".json"):
            with open(f"{path}/{runfile}", "r") as target_file:
                run_parameters.append(json.load(target_file))
    return run_parameters


def create_test_folder(path: str, run_parameters: dict) -> str:
    """Creates a folder in the specified location and saves the parameter file in it.

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
        json.dump(run_parameters, target_file)

    return run_folder_path


def generate_network(network_parameters: dict, save_path: str) -> nx.Graph:
    """Generates a new network with the given parameters and saves it in save file."""
    model: str = network_parameters["model"]
    num_nodes: int = network_parameters["num_nodes"]

    logger.debug(f"Generating new {model} network with {num_nodes} nodes.")

    if model == "albert-barabasi":
        num_attachments: int = network_parameters["num_attachments"]
        network = ng.BarabasiAlbertGenerator(num_nodes, num_attachments)()
        network.name = f"albert-barabasi_{num_nodes}n_{num_attachments}a"
    else:
        raise ValueError(f"Unknown network model: {model}")

    network_id: str = network_parameters["network_id"]

    nx.write_graphml(network, f"{save_path}{network_id}")
    logger.debug(f"Saved network to {save_path}{network_id}")
    return network


def load_network(save_path: str, network_id: str) -> nx.Graph:
    """Loads the network with the network_id in the specified save path."""
    logger.debug(f"Loading network from {save_path}/{network_id}")
    return nx.read_graphml(f"{save_path}/{network_id}")


def setup_network(network_parameters: dict, work_path: str, archive_path: str) -> nx.Graph:
    generate_new: bool = network_parameters["generate_new"]

    if generate_new is False:
        network_id: str = network_parameters["network_id"]
        network = load_network(archive_path, network_id)
    else:
        network = generate_network(network_parameters, work_path)
        network_id: str = network.name

    logger.info(f"Network {network_id} of type {network.name} and with {network.number_of_nodes()} nodes setup")
    return network


def run_queue(run_files: list[dict], save_path: str, archive_path: str) -> None:
    """Runs the tests specified in the runfiles. The results will be saved in save_path.

    Parameters
    ----------
    run_files : (list[dict])
    A list of dictionaries that contain the parameters for the tests.

    save_path : (str)
    The path to the folder where the results should be saved.

    archive_path : (str)
    The path to the archive. Only needed if network is loaded from existing ones.

    The results will be saved in save_path in folders named by the run_id

    Returns None"""
    run_times: list[float] = []
    run_ids: list[str] = []
    logger.info(f"Started {len(run_files)} runs.")

    for run_parameters in run_files:
        run_id: str = run_parameters["run_id"]
        logger.info(f"Started run_id: {run_id}")

        start_time = time.time()

        work_path: str = create_test_folder(save_path, run_parameters)

        network_parameters: dict = run_parameters["network"]
        network = setup_network(network_parameters, archive_path)

        try:
            compute_run(network, run_parameters, work_path)

        except Exception as err:
            end_time = time.time()
            run_time = end_time - start_time
            logger.error(f"An Exception occurred in run: {run_id} after {run_time}\nException: {str(err)}\n")
            err.add_note(f"Occurred in run: {run_id}")

        else:
            end_time = time.time()
            run_times.append(end_time - start_time)
            logger.info(f"Finished run: {run_id} without Exceptions! in {run_times[-1]} seconds.\n")
            run_ids.append(run_id)

            with open(f"{work_path}run_finished.txt", "w") as file:
                file.write(f"The existence of this file indicates, that the run {run_id} finished without errors.")
            logger.debug("run_finished file created.")

    logger.info(f"Finished runs : {run_ids} in {sum(run_times)} seconds.\n\n")
    return

