from .run_method import *
import logging

logger = logging.getLogger("cv_testing.compute")


def save_misc_info(path: str, data: dict) -> None:
    with open(f"{path}misc_data.txt", "a") as file:
        for key, value in data.items():
            file.write(f"{key}: {value}\n")
    return


def compute_run(network, parameters: dict, work_path: str):

    runlog_handler = logging.FileHandler(f"{work_path}/runlog.log")
    runlog_handler.setLevel(logging.DEBUG)
    runlog_formatter = logging.Formatter("%(asctime)s - %(message)s")
    runlog_handler.setFormatter(runlog_formatter)
    logger.addHandler(runlog_handler)

    try:
        dynamic_parameters: dict = parameters["dynamic"]
        dynamic = setup_dynamic(dynamic_parameters, network)

        simulation_parameters: dict = parameters["simulation"]

        sampling_parameters: dict = simulation_parameters["sampling"]
        anchors, samples = sample_anchors(dynamic, sampling_parameters, work_path)

        dim_estimate, transition_manifold = approximate_tm(dynamic, samples, work_path)
        save_misc_info(work_path, {"dimension_estimate": dim_estimate})

        linear_regression(simulation_parameters, transition_manifold, anchors, dynamic, work_path)
    except Exception as e:
        logger.error(str(e))
        raise e
    return
