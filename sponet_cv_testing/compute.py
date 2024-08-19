import numpy as np
import networkx as nx
import logging

import sponet_cv_testing.resultmanagement as rm
from sponet_cv_testing.computation.run_method import (
    setup_dynamic,
    create_anchor_points,
    sample_anchors,
    approximate_transition_manifolds,
    linear_regression
)

logger = logging.getLogger("testpipeline.compute")


def compute_run(network: nx.Graph,
                parameters: dict,
                result_path: str,
                save_samples: bool = True) -> None:
    """
    High level function for computing a cv run. This function calls individual computation function and handles
    parameter unpacking and result saving.

    Parameters
    ----------
    network : nx.Graph
        Network on which the dynamics will be simulated.
    parameters : dict
        Parameters in dict format as specified in runfile_doc.md
    result_path : str
        Path to the directory where results will be saved.
    save_samples : bool
        If set to False, the samples of the dynamics used for computing the transition manifold will not be saved.
        These samples make up the majority of the disk space.

    Returns
    -------
    None

    """

    runlog_handler = logging.FileHandler(f"{result_path}/runlog.log")
    runlog_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    runlog_handler.setLevel(logging.DEBUG)
    runlog_handler.setFormatter(runlog_formatter)
    logger.addHandler(runlog_handler)

    try:
        dynamic_parameters: dict = parameters["dynamic"]
        simulation_parameters: dict = parameters["simulation"]
        sampling_parameters: dict = simulation_parameters["sampling"]

        dynamic = setup_dynamic(dynamic_parameters, network)

        num_nodes = dynamic.num_agents
        num_coordinates = simulation_parameters["num_coordinates"]
        triangle_speedup = simulation_parameters["triangle_speedup"]
        num_opinions = dynamic_parameters["num_states"]

        # Sampling Parameters
        lag_time = sampling_parameters["lag_time"]
        short_integration_time = sampling_parameters.get("short_integration_time", -1)
        num_time_steps = sampling_parameters.get("num_timesteps", 1)
        num_anchor_points = sampling_parameters["num_anchor_points"]
        num_samples_per_anchor = sampling_parameters["num_samples_per_anchor"]

        if dynamic.num_opinions <= 256:
            state_type = np.uint8
        elif dynamic.num_opinions <= 65535:
            state_type = np.uint16
        else:
            state_type = np.uint32

        logger.info(f"Creating {num_anchor_points} anchor points.")
        anchors = create_anchor_points(dynamic,
                                       num_anchor_points,
                                       lag_time,
                                       short_integration_time
                                       )

        rm.save_anchor_points(result_path, anchors.astype(state_type))
        logger.info(f"Sampling {num_samples_per_anchor} samples per {num_anchor_points} "
                    f"with {num_time_steps} time steps.")
        samples = sample_anchors(dynamic,
                                 anchors,
                                 lag_time,
                                 num_time_steps,
                                 num_samples_per_anchor
                                 )
        if save_samples:
            rm.save_network_dynamics_samples(result_path, samples.astype(state_type))

        logger.info(f"Computing {num_time_steps} diffusion maps.")
        (diffusion_maps,
         diffusion_maps_eigenvalues,
         bandwidth_diffusion_maps,
         dimension_estimates,
         distance_matrices_compute_times
         ) = (
            approximate_transition_manifolds(samples,
                                             num_nodes,
                                             num_coordinates,
                                             triangle_speedup)
        )
        rm.save_transition_manifold(result_path, diffusion_maps)
        rm.save_diffusion_maps_eigenvalues(result_path, diffusion_maps_eigenvalues)
        rm.save_dimension_estimate(result_path, dimension_estimates)


        rm.save_compute_times(result_path,
                              distance_matrices_compute_times,
                              f"triangle_inequality_speedup={triangle_speedup}")

        cv_coefficients, cv_samples, cv, cv_coefficients_weighted, cv_samples_weighted, cv_weighted = (
            linear_regression(diffusion_maps,
                              anchors,
                              network,
                              num_opinions)
        )

        rm.save_cv_coefficients(result_path, cv_coefficients)
        rm.save_cv_samples(result_path, cv_samples)
        rm.save_cv(result_path, cv)

        rm.save_cv_coefficients_weighted(result_path, cv_coefficients_weighted)
        rm.save_cv_samples_weighted(result_path, cv_samples_weighted)
        rm.save_cv_weighted(result_path, cv_weighted)

    except Exception as e:
        logger.debug(str(e))
        logger.removeHandler(runlog_handler)
        raise e
    logger.removeHandler(runlog_handler) 
    # Handler has to be removed at the end to avoid duplicate logging of consecutive runs
    return
