import json
import os
import shutil
import datetime as dt
import numpy as np
import networkx as nx



def create_result_dir(
        path: str,
        run_parameters: dict,
        network: nx.Graph,
        overwrite: bool=False
) -> str:
    """
    Creates a directory with needed subdirectories in the specified path and creates some metadata files.

    Parameters
    ----------
    path : str
        The path in which the folder will be created.
    run_parameters : dict
        Dictionary of parameters as specified in runfile_doc.md.
    network : nx.Graph
        Network which will be saved in the directory.
    overwrite : bool, default=False
        If set to True, already existing folders with the same run_id will be overwritten.

    Returns
    -------
    str
        Path to the created folder.
    """
    run_id: str = run_parameters["run_id"]
    run_folder_path: str = f"{path}/{run_id}/"

    if os.path.isdir(run_folder_path) and overwrite:
        shutil.rmtree(run_folder_path)

    os.mkdir(run_folder_path)
    os.mkdir(f"{run_folder_path}misc_data/")

    remarks = run_parameters.get("remarks", "")

    with open(f"{run_folder_path}misc_data/remarks.txt", "w") as remarks_file:
        remarks_file.write(remarks)

    with open(f"{run_folder_path}parameters.json", "w") as target_file:
        json.dump(run_parameters, target_file, indent=3)

    save_network(network, f"{run_folder_path}", "network")

    now = dt.datetime.now().strftime("%d.%m.%Y %H:%M")
    write_metadata(run_folder_path, {"run_started": now})

    os.makedirs(f"{run_folder_path}samples/", exist_ok=True)
    os.makedirs(f"{run_folder_path}transition_manifolds/", exist_ok=True)
    os.makedirs(f"{run_folder_path}collective_variables/", exist_ok=True)
    os.makedirs(f"{run_folder_path}misc_data/", exist_ok=True)

    return run_folder_path


def write_metadata(run_folder_path: str, data: dict) -> None:
    with open(f"{run_folder_path}/misc_data/meta_data.txt", "a") as file:
        for key, value in data.items():
            file.write(f"{key}:{str(value)}\n")
    return


def create_finished_file(result_folder_path: str, run_id) -> None:
    with open(f"{result_folder_path}run_finished.txt", "w") as file:
        file.write(f"The existence of this file indicates, that the run {run_id} finished without errors."
                   f"\ntime: {dt.datetime.now().strftime('%d.%m.%Y %H:%M')}")


def save_compute_times(path: str, times: np.ndarray, header="") -> None:
    times = np.array([str(dt.timedelta(seconds=t)) for t in times])
    np.savetxt(f"{path}misc_data/distance_matrices_compute_time", times, fmt="%s", header=header)
    return


def open_network(path: str, network_id: str) -> nx.Graph:
    return nx.read_graphml(f"{path}{network_id}")


def save_network(network: nx.Graph, save_path: str, filename: str) -> None:
    nx.write_graphml(network, f"{save_path}{filename}")
    return


def get_parameters(result_dir_path: str) -> dict:
    with open(f"{result_dir_path}parameters.json", "r") as file:
        run_params = json.load(file)
    return run_params


def get_result_format(result_dir_path: str) -> str:
    file_list = os.listdir(result_dir_path)
    if "x_data.npz" in file_list:
        return "old"
    return "new"





def get_anchor_points(result_dir_path: str) -> np.ndarray:
    version = get_result_format(result_dir_path)
    if version == "old":
        return np.load(f"{result_dir_path}x_data.npz")["x_anchor"]
    return np.load(f"{result_dir_path}samples/network_anchor_points.npy")

def save_anchor_points(result_path: str, anchors: np.ndarray) -> None:
    np.save(f"{result_path}samples/network_anchor_points", anchors)
    return


def get_network_dynamics_samples(result_dir_path: str) -> np.ndarray:
    version = get_result_format(result_dir_path)
    if version == "old":
        return np.load(f"{result_dir_path}x_data.npz")["x_samples"]
    return np.load(f"{result_dir_path}samples/network_dynamics_samples.npy")

def save_network_dynamics_samples(
        result_path: str,
        samples: np.ndarray,
        save_separate: bool=False
) -> None:
    if save_separate:
        num_time_steps = samples.shape[2]
        for i in range(num_time_steps):
            np.save(f"{result_path}samples/network_dynamics_samples_{i}", samples)
        return
    np.save(f"{result_path}samples/network_dynamics_samples", samples)
    return


def get_transition_manifold(result_dir_path: str) -> np.ndarray:
    version = get_result_format(result_dir_path)
    if version == "old":
        return np.array([np.load(f"{result_dir_path}/transition_manifold.npy")])
    return np.load(f"{result_dir_path}transition_manifolds/transition_manifolds.npy")

def save_transition_manifold(result_path: str, transition_manifold: np.ndarray) -> None:
    np.save(f"{result_path}transition_manifolds/transition_manifolds", transition_manifold)
    return


def save_diffusion_maps_eigenvalues(result_path: str, eigenvalues: np.ndarray) -> None:
    np.save(f"{result_path}transition_manifolds/diffusion_maps_eigenvalues", eigenvalues)
    return


def get_dimension_estimate(result_dir_path: str) -> np.ndarray:
    version = get_result_format(result_dir_path)
    if version == "old":
        return np.array([])
    return np.load(f"{result_dir_path}transition_manifolds/intrinsic_dimension_estimates.npy")

def save_dimension_estimate(result_path: str, dimension_estimates: np.ndarray) -> None:
    np.save(f"{result_path}transition_manifolds/intrinsic_dimension_estimates", dimension_estimates)
    return


def get_cv_coefficients(result_dir_path: str) -> np.ndarray:
    version = get_result_format(result_dir_path)
    if version == "old":
        return np.array([np.load(f"{result_dir_path}cv_optim.npz")["alphas"]])
    return np.load(f"{result_dir_path}collective_variables/cv_coefficients.npy")

def save_cv_coefficients(result_path: str, cv_coeffs: np.ndarray) -> None:
    np.save(f"{result_path}collective_variables/cv_coefficients", cv_coeffs)
    return


def get_cv_coefficients_weighted(result_dir_path: str) -> np.ndarray:
    version = get_result_format(result_dir_path)
    if version == "old":
        return np.array([np.load(f"{result_dir_path}cv_optim_degree_weighted.npz")["alphas"]])
    return np.load(f"{result_dir_path}collective_variables/cv_coefficients_weighted.npy")

def save_cv_coefficients_weighted(result_path: str, cv_coeffs) -> None:
    np.save(f"{result_path}collective_variables/cv_coefficients_weighted", cv_coeffs)
    return


def get_cv_samples(result_dir_path: str) -> np.ndarray:
    version = get_result_format(result_dir_path)
    if version == "old":
        return np.array([np.load(f"{result_dir_path}cv_optim.npz")["xi_fit"]])
    return np.load(f"{result_dir_path}collective_variables/cv_samples.npy")

def save_cv_samples(result_path: str, samples: np.ndarray) -> None:
    np.save(f"{result_path}collective_variables/cv_samples", samples)


def get_cv_samples_weighted(result_dir_path: str) -> np.ndarray:
    version = get_result_format(result_dir_path)
    if version == "old":
        return np.array([np.load(f"{result_dir_path}cv_optim_degree_weighted.npz")["xi_fit"]])
    return np.load(f"{result_dir_path}collective_variables/cv_samples_weighted.npy")

def save_cv_samples_weighted(result_path: str, samples: np.ndarray) -> None:
    np.save(f"{result_path}collective_variables/cv_samples_weighted", samples)


def save_cv(result_path: str, cv) -> None:
    np.save(f"{result_path}/collective_variables/cv", cv)
    return


def save_cv_weighted(result_path: str, cv) -> None:
    np.save(f"{result_path}/collective_variables/cv_weighted", cv)

