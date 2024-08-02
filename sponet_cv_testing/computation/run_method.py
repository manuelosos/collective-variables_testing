import logging
import pickle
from collections.abc import Iterable
import networkx as nx
import numpy as np
import sponet.states
from sponet import CNVMParameters, CNTMParameters
from sponet.multiprocessing import sample_many_runs
from numba import njit, prange

from sponet_cv_testing.computation.interpretable_cvs import (
    optimize_fused_lasso,
    build_cv_from_alpha,
    compute_transition_manifold
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
        simulation_parameters: dict,
        x_anchor: np.ndarray
):
    num_anchor_points: int = sampling_parameters["num_anchor_points"]
    num_samples_per_anchor: int = sampling_parameters["num_samples_per_anchor"]
    lag_time: float = sampling_parameters["lag_time"]
    num_timesteps: int = simulation_parameters.get("num_timesteps", 1)
    if num_timesteps < 1:
        num_timesteps = 1

    logger.debug(f"Simulating voter model on {num_anchor_points} anchors")
    t, x_samples = sample_many_runs(dynamic, x_anchor, lag_time, num_timesteps+1, num_samples_per_anchor)
    x_samples = x_samples[:, :, 1:, :]

    return x_samples

#@njit(parallel=True)
def approximate_tm(
        samples: np.ndarray,
        num_nodes: int,
        num_coordinates: int,
        triangle_speedup: bool,
        simulation_parameters: dict,

) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    num_timesteps = samples.shape[2]
    num_anchorpoints: int = samples.shape[0]

    sigma = (num_nodes / 2) ** 0.5

    if triangle_speedup:

        diffusion_maps = np.empty((num_timesteps, num_anchorpoints, num_coordinates))
        diffusion_maps_eigenvalues = np.empty((num_timesteps, num_coordinates))
        dimension_estimates = np.empty(num_timesteps)
        bandwidth_diffusion_maps = np.empty(num_timesteps)

        for i in range(num_timesteps):
            (diffusion_maps[i, :, :], diffusion_maps_eigenvalues[i, :],
             bandwidth_diffusion_maps[i], dimension_estimates[i]) = (
                compute_transition_manifold(samples[:, :, i, :],
                                            sigma,
                                            num_coordinates,
                                            distance_matrix_triangle_inequality_speedup=triangle_speedup))

        logger.info(f"Approximating transition manifold with dimension={num_coordinates}")

    else:
        diffusion_maps, diffusion_maps_eigenvalues, dimension_estimates, bandwidth_diffusion_maps = (
            _parallel_transition_manifold_triangle_inequality(samples, sigma, num_coordinates))

    return diffusion_maps, diffusion_maps_eigenvalues, bandwidth_diffusion_maps, dimension_estimates


@njit(parallel=True)
def _parallel_transition_manifold_triangle_inequality(samples: np.ndarray, sigma: float, num_coordinates: int):

    num_anchorpoints = samples.shape[0]
    num_timesteps = samples.shape[2]

    diffusion_maps = np.empty((num_timesteps, num_anchorpoints, num_coordinates))
    diffusion_maps_eigenvalues = np.empty((num_timesteps, num_coordinates))
    bandwidth_diffusion_maps = np.empty(num_timesteps)
    dim_estimates = np.empty(num_timesteps)

    for i in prange(samples.shape[2]):
        diffusion_maps[i, :, :], diffusion_maps_eigenvalues[i, :], bandwidth_diffusion_maps[i], dim_estimates[i] = (
            compute_transition_manifold(samples[:, :, i, :],
                                        sigma,
                                        num_coordinates,
                                        distance_matrix_triangle_inequality_speedup=True))
    return diffusion_maps, diffusion_maps_eigenvalues, bandwidth_diffusion_maps, dim_estimates


def _sequential_transition_manifolds(dynamic, simulation_parameters, samples):
    num_timesteps = samples.shape[2]
    num_anchorpoints = samples.shape[0]
    num_coordinates = simulation_parameters["num_coordinates"]



    return



def linear_regression(
        parameters: dict,
        transition_manifold: np.ndarray,
        anchors: np.ndarray,
        dynamic: CNVMParameters
):
    num_coordinates = parameters["num_coordinates"]

    xi = transition_manifold[:, :num_coordinates]
    network = dynamic.network

    # no pre-weighting
    pen_vals = np.logspace(3, -2, 6)
    logger.info(f"Starting linear regression without pre weighting\nTested penalizing values: {pen_vals}")
    alphas, colors = optimize_fused_lasso(anchors, xi, network, pen_vals, performance_threshold=0.999)

    xi_cv = build_cv_from_alpha(alphas, dynamic.num_opinions)

    # pre-weighting
    weights = np.array([d for _, d in network.degree()])
    pen_vals = np.logspace(3, -2, 6)
    logger.info(f"Starting linear regression with pre weighting\nTested penalizing values: {pen_vals}")

    alphas_weighted, colors_weighted = optimize_fused_lasso(
        anchors, xi, network, pen_vals, weights=weights, performance_threshold=0.999
    )

    xi_cv_weighted = build_cv_from_alpha(alphas, dynamic.num_opinions, weights=weights)

    return alphas, colors, xi_cv, alphas_weighted, colors_weighted, xi_cv_weighted
