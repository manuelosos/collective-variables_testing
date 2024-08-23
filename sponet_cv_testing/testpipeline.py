import os
import json
import time
import logging
import networkx as nx
import sys
import datetime as dt
import numba

from sponet import network_generator as ng
from sponet_cv_testing.compute import compute_run
import sponet_cv_testing.resultmanagement as rm

logger = logging.getLogger("testpipeline.runmanagement")


def run_queue(
        runfile_path: str,
        result_path: str,
        network_path: str | None = None,
        num_threads: int | None = None,
        delete_samples: bool = False,
        error_exit: bool = False,
        delete_runfiles: bool = False,
        overwrite_results: bool = False,
        misc_data: dict = None
) -> None:
    """
    Computes the runs specified in the runfiles.

    The runs are specified in .json files whose structure is described in "runfile_doc.md".
    The results of the runs are saved in individual directories located in result_path.
    The result directories will be named after the run_id of the corresponding run.

    Parameters
    ----------
    runfile_path : str
        Path to the runfiles.
        If only one run should be executed, path has to end with .json.
        If multiple runs should be queued, path must lead to the directory which contains the runfiles.

    result_path : str
        Path to a directory where the results will be saved.

    network_path : str
        Path to a directory from where the networks are loaded.

    num_threads : int, default=None
        Number of Numba generated threads that are available for computation.
        If not set, the number of available threads will be chosen automatically by the Numba library.
        If Hyper Threading (or a similar concept) is enabled on the device, Numba may choose a higher number of threads
        then there are physical cores which leads to a slower computation.

    delete_samples : bool, default=False
        If set to True, the dynamics samples will not be saved in the results-directory.
        Set this to true if you have limited disk space.

    delete_runfiles : bool, default=False
        If set to True, the runfiles in the runfile folder will be deleted after successful execution.

    error_exit : bool : bool, default=False
        If set to True, the runqueue will exit with return value 1 after an error occurred.
        This implies in extension, that runs that the runqueue will not continue after the error.

    overwrite_results : bool, default=False
        Set to True to overwrite already existing results.
        If not set to True, an Error will be raised if results with equal run_id already exist in the result_dir_path

    misc_data : dict, default=None
        Miscellaneous data that will be saved in the metadata of the run.

    Returns
    -------
    None

    """

    run_files = get_runfiles(runfile_path)

    run_times: list[float] = []
    run_ids: list[str] = []
    logger.info(f"Started run-queue with {len(run_files)} runs.")

    if num_threads:
        numba.set_num_threads(num_threads)

    for run_parameters in run_files:

        run_id: str = run_parameters["run_id"]
        logger.info(f"Started run_id: {run_id}")
        start_time = time.time()

        network_parameters: dict = run_parameters["network"]

        network = setup_network(network_parameters, network_path)

        run_result_path: str = rm.create_result_dir(result_path, run_parameters, network, overwrite_results)

        rm.write_metadata(run_result_path, misc_data)

        try:
            compute_run(network,
                        run_parameters,
                        run_result_path,
                        save_samples=not delete_samples)

        except Exception as err:
            end_time = time.time()
            run_time = end_time - start_time
            logger.error(f"An Exception occurred in run: {run_id} after {run_time}\nException: {str(err)}\n")
            with open(f"{run_result_path}ERROR.txt", "w") as file:
                file.write(str(err))
            if error_exit:
                raise (err)
                sys.exit(1)
        else:
            end_time = time.time()
            run_time = end_time - start_time
            run_times.append(run_time)
            rm.write_metadata(run_result_path, {"run_time": dt.timedelta(seconds=run_time)})

            logger.info(f"Finished run: {run_id} without Exceptions! in {run_times[-1]} seconds.\n")
            run_ids.append(run_id)

            rm.create_finished_file(run_result_path, run_id)

            if delete_runfiles:
                delete_runfile(runfile_path, run_id)

            logger.debug("run_finished file created.")

    logger.info(f"Finished runs : {run_ids} in {dt.timedelta(seconds=sum(run_times))}\n\n")
    return


def get_runfiles(path: str) -> list[dict]:
    """
    Reads the json file(s) in path.

    If path points to a directory, all json files in this directory will be returned.
    Path can also point to a single json file.
    In this case a list containing the single specified file is returned.

    Parameters
    ----------
    path : str
        Path to the folder containing the json file or to a single json file itself.
        If a directory is specified the path must end with "/".
        Otherwise, the path mus end with ".json".

    Returns
    -------
    list[dict]
        List of dictionaries containing the parsed json files.
    """

    if path.endswith('/'):
        files = os.listdir(path)
        file_contents = []
        for file in files:
            if file.endswith(".json"):
                with open(f"{path}{file}", "r") as target_file:
                    file_contents.append(json.load(target_file))

    elif path.endswith(".json"):
        with open(path, "r") as target_file:
            file_contents = [json.load(target_file)]

    else:
        raise FileNotFoundError(f"Path {path} is not valid!")

    return file_contents


def delete_runfile(path: str, run_id: str) -> None:
    """
    Searches for the runfile in path with the specified run_id and deletes it.

    Parameters
    ----------
    path : str
        Path to the folder containing the runfile with the specified run_id.
    run_id : str
        Run id if the run whose runfile should be deleted.
    Returns
    -------
    None
    """
    if path.endswith('/'):
        files = os.listdir(path)
        for file in files:
            if file.endswith(".json"):
                with open(f"{path}{file}", "r") as target_file:
                    this_run_id = json.load(target_file)["run_id"]
                if this_run_id == run_id:
                    os.remove(f"{path}{file}")

    elif path.endswith(".json"):
        os.remove(path)

    return


def setup_network(network_parameters: dict, network_dir_path: str) -> nx.Graph:
    """
    Sets up the network with the given parameters.

    Parameters
    ----------
    network_parameters : dict
        Network Parameters in dict format as specified in runfile_doc.md.
        Depending on the parameters specified in network_parameters,
        the network will be generated new or an existing one will be loaded.
    network_dir_path : str
        Path to where already existing networks are located.

    Returns
    -------
        nx.Graph
    """
    generate_new: bool = network_parameters["generate_new"]

    if generate_new is False:
        network_id: str = network_parameters["network_id"]
        network = rm.open_network(network_dir_path, network_id)
    else:
        network = generate_network(network_parameters)

    return network


def generate_network(network_parameters: dict) -> nx.Graph:
    """
    Generates a new network with the given parameters.

    Parameters
    ----------
    network_parameters : dict
        Parameters in dict format as specified in runfile_doc.md.

    Returns
    -------
    nx.Graph
    """
    model: str = network_parameters["model"]
    num_nodes: int = network_parameters["num_nodes"]

    if model == "albert-barabasi":
        num_attachments: int = network_parameters["num_attachments"]
        network = ng.BarabasiAlbertGenerator(num_nodes, num_attachments)()
        network.name = "alber-barabasi"

    elif model == "holme-kim":
        num_attachments: int = network_parameters["num_attachments"]
        triad_probability: float = network_parameters["triad_probability"]
        network = nx.powerlaw_cluster_graph(num_nodes, num_attachments, triad_probability)
        network.name = "holme-kim"
    else:
        raise ValueError(f"Unknown network model: {model}")

    return network











