import logging
from collections.abc import Iterable
import networkx as nx
import numpy as np
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

    x_anchor = sample_states_local_clusters(
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

    print(num_time_steps)
    print(x_samples.shape)

    return x_samples


def approximate_transition_manifolds(
        samples: np.ndarray,
        num_nodes: int,
        num_coordinates: int,
        distance_matrix_triangle_inequality_speedup: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    num_time_steps = samples.shape[2]
    logger.debug(f"Computing {num_time_steps} distance_matrices "
                 f"with triangle speedup = {distance_matrix_triangle_inequality_speedup}.")
    bandwidth_transitions = (num_nodes / 2) ** 0.5

    distance_matrices, distance_matrix_compute_times = (
        compute_distance_matrices(samples, bandwidth_transitions, distance_matrix_triangle_inequality_speedup)
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
        logger.debug(f"Time step {i}: Starting linear regression without pre weighting.")
        arr_alphas[i, :, :], arr_colors[i, :, :] = optimize_fused_lasso(anchors,
                                                                        transition_manifold_samples[i],
                                                                        network,
                                                                        pen_vals,
                                                                        performance_threshold=0.999
                                                                        )

        xi_cvs.append(build_cv_from_alpha(arr_alphas[i, :, :], num_opinions))

        # with pre-weighting
        logger.debug(f"Time step {i}: Starting linear regression with pre weighting.")
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


def sample_states_local_clusters(
    network: nx.Graph,
    num_opinions: int,
    num_states: int,
    max_num_seeds: int = 1,
    min_num_seeds: int = 1,
    rng= np.random.default_rng(),
) -> np.ndarray:
    """
    Create states by the following procedure:
    1) Pick uniformly random opinion shares
    2) Pick num_seeds random seeds on the graph for each opinion
    (num_seeds is uniformly random between min_num_seeds and max_num_seeds)
    3) Propagate the opinions outward from each seed to neighboring nodes until the shares are reached

    Parameters
    ----------
    network : nx.Graph
    num_opinions : int
    num_states : int
    max_num_seeds : int, optional
    min_num_seeds : int, optional
    rng : Generator, optional
        random number generator

    Returns
    -------
    np.ndarray
    """
    num_agents = network.number_of_nodes()
    x = np.zeros((num_states, num_agents))
    alpha = np.ones(num_opinions)

    for i in range(num_states):
        target_shares = rng.dirichlet(alpha=alpha)
        target_counts = np.round(target_shares * num_agents).astype(int)
        target_counts[-1] = num_agents - np.sum(target_counts[:-1])
        this_x = -1 * np.ones(num_agents)  # -1 stands for not yet specified
        counts = np.zeros(num_opinions)  # keep track of current counts for each opinion

        # pick initial seeds
        num_seeds = rng.integers(min_num_seeds, max_num_seeds + 1)
        seeds = rng.choice(num_agents, size=num_seeds * num_opinions, replace=False)
        rng.shuffle(seeds)
        seeds = seeds.reshape(
            (num_opinions, num_seeds)
        )  # keep track of seeds of each opinion
        seeds = list(seeds)

        counts_reached = np.zeros(num_opinions).astype(bool)

        while True:
            # iterate through seeds and propagate opinions
            opinions = np.array(range(num_opinions))
            rng.shuffle(opinions)
            for m in opinions:
                # if counts are reached, there is nothing to do
                if counts_reached[m]:
                    continue

                # if there are no seeds available, add a random new one
                if len(seeds[m]) == 0:
                    possible_idx = np.nonzero(this_x == -1)[0]
                    new_seed = rng.choice(possible_idx)
                    seeds[m] = np.array([new_seed])

                new_seeds_m = []
                # set opinion of seeds to m
                for seed in seeds[m]:
                    if this_x[seed] != -1:
                        continue

                    if counts[m] < target_counts[m]:
                        this_x[seed] = m
                        counts[m] += 1

                        # add neighbors that are available as new seeds
                        neighbors = np.array([n for n in network.neighbors(seed)])
                        neighbors = neighbors[this_x[neighbors] == -1]
                        new_seeds_m += neighbors.tolist()

                    if counts[m] == target_counts[m]:  # counts have been reached
                        counts_reached[m] = True
                        break

                new_seeds_m = np.unique(new_seeds_m)
                rng.shuffle(new_seeds_m)
                seeds[m] = new_seeds_m

            if np.all(counts_reached):
                break

        x[i] = this_x

    x = np.unique(x.astype(int), axis=0)

    while x.shape[0] != num_states:
        missing_points = num_states - x.shape[0]
        x = np.concatenate(
            [
                x,
                sample_states_local_clusters(
                    network,
                    num_opinions,
                    missing_points,
                    max_num_seeds,
                    min_num_seeds,
                    rng,
                ),
            ]
        )
        x = np.unique(x.astype(int), axis=0)

    return x
