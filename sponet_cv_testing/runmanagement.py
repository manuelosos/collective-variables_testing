import os
import json
import time
import logging
import networkx as nx
import sys
import datetime
from sponet import network_generator as ng
from sponet_cv_testing.compute import compute_run
import workdir

import sponet_cv_testing.datamanagement as dm

logger = logging.getLogger("testpipeline.runmanagement")


def get_runfiles(path: str) -> list[dict]:
    """
    Reads the json file(s) in path.
    If path points to a directory, all json files in this dir will be read.
    Path can also point to a single json file.
    In this case a list containing the single specified file is returned.

    Parameters

    path (str) :
        Path to the folder containing the json file or to a single json file itself.
        If a directory is specified the path must end with "/".
        Example: the json files are in the directory "test_files" then "[..]/test_files/" must be passed as path.

    Returns  (list[dict])
    list of dictionaries containing the parsed json files
    """

    if path.endswith('/'):
        runfiles = os.listdir(path)
        run_parameters = []
        for runfile in runfiles:
            if runfile.endswith(".json"):
                with open(f"{path}{runfile}", "r") as target_file:
                    run_parameters.append(json.load(target_file))

    elif path.endswith(".json"):
        with open(path, "r") as target_file:
            run_parameters = [json.load(target_file)]

    else:
        raise FileNotFoundError(f"Path {path} is not valid!")

    return run_parameters


def generate_network(network_parameters: dict, save_path: str, filename: str="network") -> nx.Graph:
    """Generates a new network with the given parameters and saves it in save file."""
    model: str = network_parameters["model"]
    num_nodes: int = network_parameters["num_nodes"]

    logger.debug(f"Generating new {model} network with {num_nodes} nodes.")

    if model == "albert-barabasi":
        num_attachments: int = network_parameters["num_attachments"]
        network = ng.BarabasiAlbertGenerator(num_nodes, num_attachments)()

    elif model == "holme-kim":
        num_attachments: int = network_parameters["num_attachments"]
        triad_probability: float = network_parameters["triad_probability"]
        network = nx.powerlaw_cluster_graph(num_nodes, num_attachments, triad_probability)

    else:
        raise ValueError(f"Unknown network model: {model}")

    dm.save_network(network, save_path, filename)
    logger.debug(f"Saved network to {save_path}{network.name}")
    return network


#TODO Workflow von schon bestehenden Netwerken überarbeiten
def setup_network(network_parameters: dict, work_path: str, archive_path: str) -> nx.Graph:
    generate_new: bool = network_parameters["generate_new"]

    if generate_new is False:
        network_id: str = network_parameters["network_id"]
        network = dm.open_network(archive_path, network_id)
    else:
        network = generate_network(network_parameters, work_path)
        network_id: str = network.name

    logger.info(f"Network {network_id} of type {network.name} and with {network.number_of_nodes()} nodes setup")
    return network


def run_queue(
        run_files: list[dict],
        save_path: str,
        archive_path: str,
        exit_after_error: bool = False,
        misc_data: dict=None
) -> None:
    """
    Runs the tests specified in the runfiles. The results will be saved in save_path.

    Parameters
    ----------
    run_files : (list[dict])
        A list of dictionaries that contain the parameters for the tests.

    save_path : (str)
        The path to the folder where the results should be saved.

    archive_path : (str)
        The path to the archive. Only needed if network is loaded from existing ones.

    exit_after_error : (bool)
        Set to true if the program should exit with system.exit(1) after an error. Default is False
        Set to true if you want to compute single file runs on cluster.

    misc_data : (dict)
        A dictionary consisting of additional data that will be saved in misc_data.txt in save_path.

    The results will be saved in save_path in folders named by the run_id

    Returns None
    """

    run_times: list[float] = []
    run_ids: list[str] = []
    logger.info(f"Started {len(run_files)} runs.")

    for run_parameters in run_files:

        run_id: str = run_parameters["run_id"]
        logger.info(f"Started run_id: {run_id}")
        start_time = time.time()

        work_path: str = workdir.create_work_dir(save_path, run_parameters)

        network_parameters: dict = run_parameters["network"]
        network = setup_network(network_parameters, work_path, archive_path)

        # TODO option zum überschreiben von schon existierenden runs einfügen.

        workdir.write_misc_data(work_path, misc_data)

        try:
            compute_run(network, run_parameters, work_path)

        except Exception as err:
            end_time = time.time()
            run_time = end_time - start_time
            logger.error(f"An Exception occurred in run: {run_id} after {run_time}\nException: {str(err)}\n")
            with open(f"{work_path}ERROR.txt", "w") as file:
                file.write(str(err))
            if exit_after_error:
                sys.exit(1)
        else:
            end_time = time.time()
            run_time = end_time - start_time
            run_times.append(run_time)
            dm.write_misc_data(work_path, {"run_time": run_time})

            logger.info(f"Finished run: {run_id} without Exceptions! in {run_times[-1]} seconds.\n")
            run_ids.append(run_id)

            with open(f"{work_path}run_finished.txt", "w") as file:
                file.write(f"The existence of this file indicates, that the run {run_id} finished without errors.")
            logger.debug("run_finished file created.")

    logger.info(f"Finished runs : {run_ids} in {sum(run_times)} seconds.\n\n")
    return

