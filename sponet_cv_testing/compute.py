from sponet_cv_testing.computation.run_method import *
import logging
import sponet_cv_testing.workdir as wd
from sponet_cv_testing.datamanagement import write_misc_data

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

        simulation_parameters: dict = parameters["simulation"]

        sampling_parameters: dict = simulation_parameters["sampling"]
        anchors = create_anchor_points(dynamic, sampling_parameters).astype(state_type)

        samples = sample_anchors(dynamic, sampling_parameters, anchors)

        np.savez_compressed(f"{work_path}x_data",
                            x_anchor=anchors,
                            x_samples=samples)

        diffusion_bandwidth, dim_estimate, transition_manifold = approximate_tm(dynamic, samples, work_path)

        wd.write_misc_data(work_path,
                           {"diffusion_bandwidth": diffusion_bandwidth,
                            "dimension_estimate": dim_estimate})

        linear_regression(simulation_parameters, transition_manifold, anchors, dynamic, work_path)
    except Exception as e:
        logger.debug(str(e))
        raise e
    return
