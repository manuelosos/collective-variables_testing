from sponet_cv_testing.computation.run_method import *
import logging

logger = logging.getLogger("testpipeline.compute")


def compute_run(network, parameters: dict, result_path: str) -> None:

    runlog_handler = logging.FileHandler(f"{result_path}/runlog.log")
    runlog_handler.setLevel(logging.DEBUG)
    runlog_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
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
                                       ).astype(state_type)
        np.save(f"{result_path}anchor_states", anchors)

        logger.info(f"Sampling {num_samples_per_anchor} samples on {num_anchor_points} anchors "
                    f"with {num_time_steps} time steps.")
        samples = sample_anchors(dynamic,
                                 anchors,
                                 lag_time,
                                 num_time_steps,
                                 num_samples_per_anchor
                                 )
        np.save(f"{result_path}samples", samples)

        logger.info(f"Computing {num_time_steps} diffusion maps")
        diffusion_maps, diffusion_maps_eigenvalues, bandwidth_diffusion_maps, dimension_estimates = (
            approximate_tm(samples,
                           num_nodes,
                           num_coordinates,
                           triangle_speedup)
        )
        np.save(f"{result_path}transition_manifolds", diffusion_maps)
        np.save(f"{result_path}diffusion_maps_eigenvalues", diffusion_maps_eigenvalues)
        np.save(f"{result_path}intrinsic_dimension_estimates", dimension_estimates)

        cv_coefficients, cv_samples, cv, cv_coefficients_weighted, cv_samples_weighted, cv_weighted = (
            linear_regression(diffusion_maps,
                              anchors,
                              network,
                              num_opinions)
        )
        np.save(f"{result_path}cv_coefficients", cv_coefficients)
        np.save(f"{result_path}cv_samples", cv_samples)
        np.save(f"{result_path}cv", cv)
        np.save(f"{result_path}cv_coefficients_weighted", cv_coefficients)
        np.save(f"{result_path}cv_samples_weighted", cv_samples)
        np.save(f"{result_path}cv_weighted", cv)

    except Exception as e:
        logger.debug(str(e))
        raise e

    return
