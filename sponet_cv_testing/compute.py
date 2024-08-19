import numpy as np
import sponet_cv_testing.resultmanagement as rm
from sponet_cv_testing.computation.run_method import *
import logging
import os
import datetime as dt

logger = logging.getLogger("testpipeline.compute")


def compute_run(network: nx.Graph,
                parameters: dict,
                result_path: str,
                delete_samples: bool = False) -> None:
    """
    High level function for computing a cv run. This function calls individual computation function and handles
    parameter unpacking and result saving.

    Parameters
    ----------
    network : nx.Graph
    parameters : dict
        See runfile_doc for more information.
    result_path : str
        Path to the directory where results will be saved.
    delete_samples : bool
        If set to True, the samples will not be saved. Samples make up the majority of storage usage.
    Returns None

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

        samples_path = f"{result_path}samples/"
        os.makedirs(samples_path, exist_ok=True)
        tm_path = f"{result_path}transition_manifolds/"
        os.makedirs(tm_path, exist_ok=True)
        cv_path = f"{result_path}collective_variables/"
        os.makedirs(cv_path, exist_ok=True)
        misc_path = f"{result_path}misc_data/"
        os.makedirs(misc_path, exist_ok=True)

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
                                       ).astype(state_type)
        np.save(f"{samples_path}network_anchor_points", anchors)

        logger.info(f"Sampling {num_samples_per_anchor} samples per {num_anchor_points} "
                    f"with {num_time_steps} time steps.")
        samples = sample_anchors(dynamic,
                                 anchors,
                                 lag_time,
                                 num_time_steps,
                                 num_samples_per_anchor
                                 )
        if not delete_samples:
            np.save(f"{samples_path}network_dynamics_samples", samples.astype(state_type))

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
        np.save(f"{tm_path}transition_manifolds", diffusion_maps)
        np.save(f"{tm_path}diffusion_maps_eigenvalues", diffusion_maps_eigenvalues)
        np.save(f"{tm_path}intrinsic_dimension_estimates", dimension_estimates)

        rm.save_compute_times(f"{misc_path}distance_matrices_compute_time",
                              distance_matrices_compute_times,
                              f"triangle_inequality_speedup={triangle_speedup}")

        cv_coefficients, cv_samples, cv, cv_coefficients_weighted, cv_samples_weighted, cv_weighted = (
            linear_regression(diffusion_maps,
                              anchors,
                              network,
                              num_opinions)
        )
        np.save(f"{cv_path}cv_coefficients", cv_coefficients)
        np.save(f"{cv_path}cv_samples", cv_samples)
        np.save(f"{cv_path}cv", cv)
        np.save(f"{cv_path}cv_coefficients_weighted", cv_coefficients)
        np.save(f"{cv_path}cv_samples_weighted", cv_samples)
        np.save(f"{cv_path}cv_weighted", cv)

    except Exception as e:
        logger.debug(str(e))
        logger.removeHandler(runlog_handler)
        raise e
    logger.removeHandler(runlog_handler) 
    # Handler has to be removed at the end to avoid duplicate logging of consecutive runs
    return
