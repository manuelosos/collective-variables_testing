import pickle
from sponet import CNVMParameters, CNTMParameters
import numpy as np
import networkx as nx
from sponet import network_generator as ng
from collections.abc import Iterable
import logging

from .interpretable_cvs import (
    TransitionManifold,
    optimize_fused_lasso,
    sample_cnvm,
    build_cv_from_alpha,
    create_anchor_points_local_clusters,
    integrate_anchor_points
)

logger = logging.getLogger("sponet_cv_testing.computation")


def generate_network(network_parameters: dict, save_path: str) -> nx.Graph:
    """Generates a new network with the given parameters and saves it in save file."""
    name: str = network_parameters["name"]
    num_nodes: int = network_parameters["num_nodes"]

    logger.debug(f"Generating new {name} network with {num_nodes} nodes.")

    if name == "albert-barabasi":
        num_attachments: int = network_parameters["num_attachments"]
        network = ng.BarabasiAlbertGenerator(num_nodes, num_attachments)()
    else:
        raise ValueError(f"Unknown network!: {name}")

    network_id: str = network_parameters["network_id"]

    nx.write_graphml(network, f"{save_path}/{network_id}")
    logger.debug(f"Saved network to {save_path}/{network_id}")
    return network


def load_network(save_path: str, network_id: str) -> nx.Graph:
    """Loads the network with the network_id in the specified save path."""
    logger.debug(f"Loading network from {save_path}/{network_id}")
    return nx.read_graphml(f"{save_path}/{network_id}")


def setup_network(network_parameters: dict, work_path: str) -> nx.Graph:
    network_id: str = network_parameters["network_id"]
    generate_new: bool = network_parameters["generate_new"]

    if generate_new is False:
        network = load_network(work_path, network_id)
    else:
        network = generate_network(network_parameters, work_path)

    logger.info(f"Network {network_id} of type {network.name} and with {network.number_of_nodes()} nodes setup")
    return network


def setup_dynamic(dynamic_parameters: dict, network: nx.Graph) -> CNVMParameters | CNTMParameters:

    num_states: int = dynamic_parameters["num_states"]
    rates: dict = dynamic_parameters["rates"]
    name: str = dynamic_parameters["name"]

    # Python lists have to be converted to numpy arrays.
    # By assumption every list passed by the json file will be converted.
    for key in rates.keys():
        if isinstance(rates[key], Iterable):
            rates[key] = np.array(rates[key])

    if name == "CNVM":
        params = CNVMParameters(
            num_opinions=num_states,
            num_agents=network.number_of_nodes(),
            network=network,
            **rates
        )
    elif name == "CNTM":
        params = CNTMParameters(
            network=network,
            **rates
        )
    else:
        raise ValueError(f"Unknown dynamic! {name}")

    logger.info(f"Dynamic of type {name} setup")

    return params


def sample_anchors(
        dynamic: CNVMParameters | CNTMParameters,
        sampling_parameters: dict,
        save_path: str
) -> tuple[np.ndarray, np.ndarray]:

    num_anchor_points: int = sampling_parameters["num_anchor_points"]
    num_samples_per_anchor: int = sampling_parameters["num_samples_per_anchor"]
    lag_time: float = sampling_parameters["lag_time"]

    logger.info(f"Starting sampling {num_anchor_points} anchors")
    x_anchor = create_anchor_points_local_clusters(
        dynamic.network, dynamic.num_opinions, num_anchor_points, 5
    )

    short_integration_time: float = lag_time/10
    logger.debug(f"Starting short integration lag_time {short_integration_time}")
    x_anchor = integrate_anchor_points(
        x_anchor, dynamic, short_integration_time
    )  # integrate shortly to get rid of unstable states

    logger.debug(f"Simulating voter model on {num_anchor_points} anchors")
    x_samples: np.ndarray = sample_cnvm(x_anchor, num_samples_per_anchor, lag_time, dynamic)

    np.savez_compressed(f"{save_path}/x_data", x_anchor=x_anchor, x_samples=x_samples)

    logger.info(f"Finished sampling")
    return x_anchor, x_samples


# TODO mit typing die Typbezeichnungen verbessern
def approximate_tm(dynamic: CNVMParameters | CNTMParameters, samples: np.ndarray, save_path: str) -> np.ndarray:

    sigma = (dynamic.num_agents / 2) ** 0.5
    d = 2

    trans_manifold = TransitionManifold(sigma, 0.1, d)

    logger.info(f"Starting approximating transition manifold with dimension={d}")
    xi = trans_manifold.fit(samples, optimize_bandwidth=True)

    np.save(f"{save_path}/transition_manifold", xi)
    logger.info(f"Finished approximating")
    return xi


def linear_regression(
        parameters: dict,
        transition_manifold: np.ndarray,
        anchors: np.ndarray,
        dynamic: CNVMParameters,
        save_path: str
):
    num_coordinates = parameters["num_coordinates"]

    xi = transition_manifold[:, :num_coordinates]
    network = dynamic.network

    # no pre-weighting
    pen_vals = np.logspace(3, -2, 6)
    logger.info(f"Starting linear regressing without pre weighting\nTested penalizing values: {pen_vals}")
    alphas, colors = optimize_fused_lasso(anchors, xi, network, pen_vals, performance_threshold=0.999)

    np.savez(f"{save_path}/cv_optim.npz", alphas=alphas, xi_fit=colors)
    logger.info("Finished linear regression")

    xi_cv = build_cv_from_alpha(alphas, dynamic.num_opinions)
    with open(f"{save_path}/cv.pkl", "wb") as file:
        pickle.dump(xi_cv, file)
    logger.info("Finished")

    # pre-weighting
    weights = np.array([d for _, d in network.degree()])
    pen_vals = np.logspace(3, -2, 6)
    logger.info(f"Starting linear regression with pre weighting\nTested penalizing values: {pen_vals}")

    alphas, colors = optimize_fused_lasso(
        anchors, xi, network, pen_vals, weights=weights, performance_threshold=0.999
    )

    np.savez(f"{save_path}/cv_optim_degree_weighted.npz", alphas=alphas, xi_fit=colors)

    xi_cv = build_cv_from_alpha(alphas, dynamic.num_opinions, weights=weights)
    with open(f"{save_path}/cv_degree_weighted.pkl", "wb") as file:
        pickle.dump(xi_cv, file)
    logger.info("Finished")

    return
