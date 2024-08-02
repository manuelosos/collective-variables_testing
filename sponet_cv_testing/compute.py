from sponet_cv_testing.computation.run_method import *
import logging
import sponet_cv_testing.workdir as wd
from sponet_cv_testing.datamanagement import write_misc_data
from numba import njit, prange

logger = logging.getLogger("testpipeline.compute")


def compute_run(network, parameters: dict, work_path: str):
    runlog_handler = logging.FileHandler(f"{work_path}/runlog.log")
    runlog_handler.setLevel(logging.DEBUG)
    runlog_formatter = logging.Formatter("%(asctime)s - %(message)s")
    runlog_handler.setFormatter(runlog_formatter)
    logger.addHandler(runlog_handler)

    try:
        dynamic_parameters: dict = parameters["dynamic"]
        dynamic = setup_dynamic(dynamic_parameters, network)

        if dynamic.num_opinions <= 256:
            state_type = np.uint8
        elif dynamic.num_opinions <= 65535:
            state_type = np.uint16
        else:
            state_type = np.uint32

    # Sampling
        simulation_parameters: dict = parameters["simulation"]

        sampling_parameters: dict = simulation_parameters["sampling"]
        anchors = create_anchor_points(dynamic, sampling_parameters).astype(state_type)

        samples = sample_anchors(dynamic, sampling_parameters, simulation_parameters, anchors)

        np.savez_compressed(f"{work_path}x_data",
                            x_anchor=anchors,
                            x_samples=samples)

        num_timesteps = simulation_parameters.get("num_timesteps", 1)

        if num_timesteps <= 1:
            xi, eigenvalues, diffusion_bandwidth, dim_estimate = (
                approximate_tm(dynamic, simulation_parameters, samples[:, :, 0, :]))

            wd.write_misc_data(work_path, {"diffusion_bandwidth": diffusion_bandwidth,
                                           "dimension_estimate": dim_estimate})

        elif simulation_parameters.get("triangle_speedup", False):
            # if no triangle speedup is enabled the transition manifolds will be computed in sequence
            _sequential_transition_manifolds(dynamic, simulation_parameters, samples)
        else:

            pass

        if num_timesteps <= 1:



            alphas, colors, xi_cv, alphas_weighted, colors_weighted, xi_cv_weighted = (
                linear_regression(simulation_parameters, xi, anchors, dynamic))

            _save_cv(work_path, alphas, colors, xi_cv)
            _save_cv_degree_weighted(work_path, alphas, colors, xi_cv_weighted)

        else:  # parallel case
            # If trianglespeedup is not enabled transition manifolds will be computed sequentially with parallel tasks
            if simulation_parameters.get("triangle_speedup", False):
                _sequential_transition_manifolds(dynamic, simulation_parameters, samples)

            _parallel_transition_matrix_and_regression(dynamic, simulation_parameters, anchors, samples, work_path)


    except Exception as e:
        logger.debug(str(e))
        raise e
    return








@njit(parallel=True)
def _parallel_transition_matrix_and_regression(dynamic, simulation_parameters: dict, anchors, samples, work_path):
    d = simulation_parameters["num_coordinates"]
    num_anchorpoints = samples.shape[0]
    num_timesteps = samples.shape[2]

    xi = np.empty((num_timesteps, num_anchorpoints, d))
    eigenvalues = np.empty((num_timesteps, d))
    diffusion_bandwidths = np.empty(num_timesteps)
    dim_estimates = np.empty(num_timesteps)

    alphas_list = np.empty((num_timesteps, dynamic.num_agents, d))
    alphas_weighted_list = np.empty((num_timesteps, dynamic.num_agents, d))
    colors_list = np.empty((num_timesteps, dynamic.num_agents))
    colors_weighted_list = np.empty((num_timesteps, dynamic.num_agents))
    xi_cv_list = [None for i in range(num_timesteps)]
    xi_cv_weighted_list = [None for i in range(num_timesteps)]

    for i in prange(samples.shape[2]):
        xi[i, :, :], eigenvalues[i, :], diffusion_bandwidths[i], dim_estimates[i] = (
            approximate_tm(dynamic, simulation_parameters, samples[:, :, i, :]))

        linear_regression(simulation_parameters, xi, anchors, dynamic)

        alphas, colors, xi_cv, alphas_weighted, colors_weighted, xi_cv_weighted = (
            linear_regression(simulation_parameters, xi, anchors, dynamic))

        alphas_list[i, :, :] = alphas
        alphas_weighted_list[i, :, :] = alphas_weighted
        colors_list[i, :] = colors
        colors_weighted_list[i, :] = colors_weighted
        xi_cv_list[i] = xi_cv
        xi_cv_weighted_list[i] = xi_cv_weighted

    _save_cv(work_path, alphas_list, colors_list, xi_cv_list)
    _save_cv_degree_weighted(work_path, alphas_weighted_list, colors_weighted_list, xi_cv_weighted_list)

    return


def _save_cv(path, alphas, colors, xi):
    np.savez(f"{path}cv_optim.npz", alphas=alphas, xi_fit=colors)
    with open(f"{path}cv.pkl", "wb") as file:
        pickle.dump(xi, file)


def _save_cv_degree_weighted(path, alphas, colors, xi):
    np.savez(f"{path}cv_optim_degree_weighted.npz", alphas=alphas, xi_fit=colors)
    with open(f"{path}cv_degree_weighted.pkl", "wb") as file:
        pickle.dump(xi, file)