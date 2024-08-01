import logging
import pickle
from collections.abc import Iterable
import networkx as nx
import numpy as np
import sponet.states
from sponet import CNVMParameters, CNTMParameters
from sponet.multiprocessing import sample_many_runs

from sponet_cv_testing.computation.interpretable_cvs import (
    TransitionManifold,
    optimize_fused_lasso,
    build_cv_from_alpha,
)

logger = logging.getLogger("testpipeline.compute")


def setup_dynamic(dynamic_parameters: dict, network: nx.Graph) -> CNVMParameters | CNTMParameters:
    num_states: int = dynamic_parameters["num_states"]
    rates: dict = dynamic_parameters["rates"]
    model: str = dynamic_parameters["model"]

    # Python lists have to be converted to numpy arrays.
    # By assumption every list passed by the json file will be converted.
    for key in rates.keys():
        if isinstance(rates[key], Iterable):
            rates[key] = np.array(rates[key])

    if model.lower() == "cnvm":
        params = CNVMParameters(
            num_opinions=num_states,
            num_agents=network.number_of_nodes(),
            network=network,
            **rates
        )
    elif model.lower() == "cntm":
        params = CNTMParameters(
            network=network,
            **rates
        )
    else:
        raise ValueError(f"Unknown dynamic! {model}")

    logger.info(f"Dynamic model {model} setup")

    return params


def create_anchor_points(
        dynamic: CNVMParameters | CNTMParameters,
        sampling_parameters: dict
):
    num_anchor_points: int = sampling_parameters["num_anchor_points"]
    num_samples_per_anchor: int = sampling_parameters["num_samples_per_anchor"]
    lag_time: float = sampling_parameters["lag_time"]
    if "short_integration_time" in sampling_parameters.keys():
        short_integration_time = sampling_parameters["short_integration_time"]
    else:
        # Set short_integration_time dependent on maximal rate
        max_rate = max(np.max(dynamic.r), np.max(dynamic.r_tilde))
        short_integration_time: float = lag_time / 10 / max_rate

    logger.info(f"Starting sampling {num_anchor_points} anchors")
    x_anchor = sponet.states.sample_states_local_clusters(
        dynamic.network, dynamic.num_opinions, num_anchor_points, 5
    )

    if short_integration_time != 0:
        logger.debug(f"Starting short integration lag_time {short_integration_time}")
        _, x = sample_many_runs(
            dynamic, x_anchor, short_integration_time, 2, 1, n_jobs=-1
        )
        x_anchor = x[:, 0, -1, :]

    return x_anchor


def sample_anchors(
        dynamic: CNVMParameters | CNTMParameters,
        sampling_parameters: dict,
        x_anchor: np.ndarray
):
    num_anchor_points: int = sampling_parameters["num_anchor_points"]
    num_samples_per_anchor: int = sampling_parameters["num_samples_per_anchor"]
    lag_time: float = sampling_parameters["lag_time"]
    num_timesteps: int = sampling_parameters.get("num_timesteps", 1)
    if num_timesteps < 1:
        num_timesteps = 1

    logger.debug(f"Simulating voter model on {num_anchor_points} anchors")
    t, x_samples = sample_many_runs(dynamic, x_anchor, lag_time, num_timesteps+1, num_samples_per_anchor)
    x_samples = x_samples[:, :, 2:, :]

    return x_samples


def approximate_tm(
        dynamic: CNVMParameters | CNTMParameters,
        simulation_parameters: dict,
        samples: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:

    sigma = (dynamic.num_agents / 2) ** 0.5
    d = simulation_parameters["num_coordinates"]

    trans_manifold = TransitionManifold(sigma, 0.1, d)
    triangle_speedup = simulation_parameters.get("triangle_speedup", False)

    logger.info(f"Approximating transition manifold with dimension={d}")

    #TODO an mehrere Timesteps anpassen

    xi = trans_manifold.fit(samples, optimize_bandwidth=True, triangle_speedup=triangle_speedup)

    bandwidth = trans_manifold.bandwidth_diffusion_map
    dimension_estimate = trans_manifold.dimension_estimate
    eigenvalues = trans_manifold.eigenvalues

    return xi, eigenvalues, bandwidth, dimension_estimate


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
    logger.info(f"Starting linear regression without pre weighting\nTested penalizing values: {pen_vals}")
    alphas, colors = optimize_fused_lasso(anchors, xi, network, pen_vals, performance_threshold=0.999)

    np.savez(f"{save_path}cv_optim.npz", alphas=alphas, xi_fit=colors)

    xi_cv = build_cv_from_alpha(alphas, dynamic.num_opinions)
    with open(f"{save_path}cv.pkl", "wb") as file:
        pickle.dump(xi_cv, file)

    # pre-weighting
    weights = np.array([d for _, d in network.degree()])
    pen_vals = np.logspace(3, -2, 6)
    logger.info(f"Starting linear regression with pre weighting\nTested penalizing values: {pen_vals}")

    alphas, colors = optimize_fused_lasso(
        anchors, xi, network, pen_vals, weights=weights, performance_threshold=0.999
    )

    np.savez(f"{save_path}cv_optim_degree_weighted.npz", alphas=alphas, xi_fit=colors)

    xi_cv = build_cv_from_alpha(alphas, dynamic.num_opinions, weights=weights)
    with open(f"{save_path}cv_degree_weighted.pkl", "wb") as file:
        pickle.dump(xi_cv, file)

    return
