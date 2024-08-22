import logging
import os
import argparse
import numpy as np

from ..sponet_cv_testing.testpipeline import get_runfiles
from ..sponet_cv_testing import resultmanagement as res
import numba
from ..sponet_cv_testing import compute as comp
from ..sponet_cv_testing.computation.run_method import (
    approximate_transition_manifolds,
    linear_regression
)




logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("index", type=int)
parser.add_argument("runfile_path",
                    help="Path to the directory where the runfiles are located.")
parser.add_argument("result_path",
                    help="Path to the directory where the results will be saved.")
parser.add_argument("network_path",
                    help="Path to the directory where the networks are located.")


def main():
    numba.set_num_threads(16)
    args = parser.parse_args()

    index = args.index
    runfile_path = args.runfile_path
    result_path = args.result_path
    network_path = args.network_path

    run_parameters = get_runfiles(runfile_path)[0]

    network = res.open_network(network_path)
    params: comp.RunParameters = comp.RunParameters(run_parameters, network)

    result_path = result_path+f"{params.run_id}/"

    anchors = res.get_anchor_points(result_path)
    samples = np.load(f"{result_path}samples/network_dynamics_samples_{index}.npy").astype(float)
    samples = np.expand_dims(samples, axis=2)
    logger.info(f"Computing {params.num_time_steps} diffusion maps.")
    (diffusion_maps,
     diffusion_maps_eigenvalues,
     bandwidth_diffusion_maps,
     dimension_estimates,
     distance_matrices_compute_times
     ) = (
        approximate_transition_manifolds(
            samples,
            params.num_nodes,
            params.num_cv_coordinates,
            params.triangle_speedup
        )
    )
    np.save(f"{result_path}transition_manifolds/transition_manifolds_{index}", diffusion_maps)
    np.save(f"{result_path}transition_manifolds/diffusion_maps_eigenvalues_{index}", diffusion_maps_eigenvalues)
    np.savetxt(
        f"{result_path}misc_data/distance_matrices_compute_time_{index}",
        distance_matrices_compute_times, fmt="%s")

    (cv_coefficients,
     cv_samples,
     cv,
     cv_coefficients_weighted,
     cv_samples_weighted,
     cv_weighted
     ) = (
        linear_regression(
            diffusion_maps,
            anchors,
            params.network,
            params.num_states
        )
    )
    np.save(f"{result_path}collective_variables/cv_coefficients_{index}", cv_coefficients)
    np.save(f"{result_path}collective_variables/cv_samples_{index}", cv_samples)
    np.save(f"{result_path}/collective_variables/cv_{index}", cv)
    np.save(f"{result_path}collective_variables/cv_coefficients_weighted_{index}", cv_coefficients_weighted)
    np.save(f"{result_path}collective_variables/cv_samples_weighted_{index}", cv_samples_weighted)
    np.save(f"{result_path}/collective_variables/cv_weighted_{index}", cv_weighted)

    os.remove(f"{result_path}samples/network_dynamics_samples_{index}.npy")


if __name__ == '__main__':
    main()