import logging
import pickle
from collections.abc import Iterable
import networkx as nx
import numpy as np
import sponet.states
from sponet import CNVMParameters, CNTMParameters
from sponet.multiprocessing import sample_many_runs

from sponet_cv_testing.computation.interpretable_cvs import (
    optimize_fused_lasso,
    build_cv_from_alpha,
    compute_diffusion_maps,
    compute_distance_matrices
)

logger = logging.getLogger("testpipeline.compute.run_method")


def setup_dynamic(
        dynamic_parameters: dict,
        network: nx.Graph
) -> CNVMParameters | CNTMParameters:
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
        num_anchor_points: int,
        lag_time: float,
        short_integration_time: float = -1,
) -> np.ndarray:
    if short_integration_time < 0:
        # Set short_integration_time dependent on maximal rate
        max_rate = max(np.max(dynamic.r), np.max(dynamic.r_tilde))
        short_integration_time: float = lag_time / 10 / max_rate

    x_anchor = sponet.states.sample_states_local_clusters(
        dynamic.network, dynamic.num_opinions, num_anchor_points, 5
    )

    if short_integration_time > 0:
        logger.debug(f"Starting short integration lag_time {short_integration_time}.")
        _, x = sample_many_runs(
            dynamic, x_anchor, short_integration_time, 2, 1, n_jobs=-1
        )
        x_anchor = x[:, 0, -1, :]

    return x_anchor


def sample_anchors(
        dynamic: CNVMParameters | CNTMParameters,
        anchors: np.ndarray,
        lag_time: float,
        num_time_steps: int,
        num_samples_per_anchor: int
) -> np.ndarray:
    if num_time_steps < 1:
        num_time_steps = 1
    t, x_samples = sample_many_runs(dynamic, anchors, lag_time, num_time_steps + 1, num_samples_per_anchor)
    x_samples = x_samples[:, :, 1:, :]

    return x_samples


def approximate_transition_manifolds(
        samples: np.ndarray,
        num_nodes: int,
        num_coordinates: int,
        distance_matrix_triangle_inequality_speedup: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_time_steps = samples.shape[2]
    logger.debug(f"Computing {num_time_steps} distance_matrices."
                 f"with triangle speedup = {distance_matrix_triangle_inequality_speedup}.")
    bandwidth_transitions = (num_nodes / 2) ** 0.5
    distance_matrices, distance_matrix_compute_times = (
        compute_distance_matrices(samples,
                                  bandwidth_transitions,
                                  distance_matrix_triangle_inequality_speedup)
    )

    logger.debug(f"Computing {num_time_steps} diffusion maps.")
    diffusion_maps, diffusion_maps_eigenvalues, dimension_estimates, bandwidth_diffusion_maps = (
        compute_diffusion_maps(distance_matrices,
                               num_coordinates))

    return (diffusion_maps, diffusion_maps_eigenvalues, bandwidth_diffusion_maps, dimension_estimates,
            distance_matrix_compute_times)


def linear_regression(
        transition_manifold_samples: np.ndarray,
        anchors: np.ndarray,
        network: nx.Graph,
        num_opinions: int
):
    num_time_steps = transition_manifold_samples.shape[0]
    num_initial_states = transition_manifold_samples.shape[1]
    num_coordinates = transition_manifold_samples.shape[2]
    num_agents = anchors.shape[1]

    arr_alphas = np.empty((num_time_steps, num_agents, num_coordinates))
    arr_colors = np.empty((num_time_steps, num_initial_states, num_coordinates))

    arr_alphas_weighted = np.empty_like(arr_alphas)
    arr_colors_weighted = np.empty_like(arr_colors)

    xi_cvs = []
    xi_cvs_weighted = []

    pen_vals = np.logspace(3, -2, 6)
    weights = np.array([d for _, d in network.degree()])

    for i in range(num_time_steps):
        # no pre-weighting
        logger.debug(f"Starting linear regression without pre weighting.")
        arr_alphas[i, :, :], arr_colors[i, :, :] = optimize_fused_lasso(anchors,
                                                                        transition_manifold_samples[i],
                                                                        network,
                                                                        pen_vals,
                                                                        performance_threshold=0.999
                                                                        )

        xi_cvs.append(build_cv_from_alpha(arr_alphas[i, :, :], num_opinions))

        # with pre-weighting
        logger.debug(f"Starting linear regression with pre weighting.")
        arr_alphas_weighted[i, :, :], arr_colors_weighted[i, :, :] = (
            optimize_fused_lasso(anchors,
                                 transition_manifold_samples[i],
                                 network,
                                 pen_vals,
                                 weights=weights,
                                 performance_threshold=0.999
                                 ))

        xi_cvs_weighted.append(build_cv_from_alpha(arr_alphas_weighted[i, :, :], num_opinions, weights=weights))

    xi_cvs = np.array(xi_cvs)
    xi_cvs_weighted = np.array(xi_cvs_weighted)

    return arr_alphas, arr_colors, xi_cvs, arr_alphas_weighted, arr_colors_weighted, xi_cvs_weighted
