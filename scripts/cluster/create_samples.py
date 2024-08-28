
import logging
import os
import argparse
from sponet_cv_testing.testpipeline import get_runfiles
import sponet_cv_testing.resultmanagement as res
import numba
import sponet_cv_testing.compute as comp
from sponet_cv_testing.computation.run_method import (
    setup_dynamic,
    create_anchor_points,
    sample_anchors,
    approximate_transition_manifolds,
    linear_regression
)




logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("runfile_path",
                    help="Path to the directory where the runfiles are located.")
parser.add_argument("result_path",
                    help="Path to the directory where the results will be saved.")
parser.add_argument("network_path",
                    help="Path to the directory where the networks are located.")



def main():

    args = parser.parse_args()

    runfile_path = args.runfile_path
    result_path = args.result_path
    network_path = args.network_path

    run_parameters = get_runfiles(runfile_path)[0]

    network = res.open_network(network_path)


    result_path = res.create_result_dir(result_path, run_parameters, network, True)



    numba.set_num_threads(16)
    misc_data: dict = {"device": "Cluster",
                       "number of numba threads": 16,
                       "cpu count": os.cpu_count(),
                       "delete_samples": False}

    params: comp.RunParameters = comp.RunParameters(run_parameters, network)


    logger.info(f"Creating {params.num_anchor_points} anchor points.")
    anchors = create_anchor_points(
        params.dynamic,
        params.num_anchor_points,
        params.lag_time,
        params.short_integration_time
    )
    res.save_anchor_points(result_path, anchors.astype(params.network_superstate_type))

    logger.info(f"Sampling {params.num_samples_per_anchor} samples per {params.num_anchor_points} "
                f"with {params.num_time_steps} time steps.")
    samples = sample_anchors(
        params.dynamic,
        anchors,
        params.lag_time,
        params.num_time_steps,
        params.num_samples_per_anchor
    )

    res.save_network_dynamics_samples(result_path, samples.astype(params.network_superstate_type), save_separate=True)

    return




if __name__ == "__main__":
    main()