from .run_method import *
import time
import logging


def compute_run(parameters: dict, work_path: str):

    network_parameters: dict = parameters["network"]
    network = setup_network(network_parameters, work_path)

    dynamic_parameters: dict = parameters["dynamic"]
    dynamic = setup_dynamic(dynamic_parameters, network)
    del network

    simulation_parameters: dict = parameters["simulation"]

    sampling_parameters: dict = simulation_parameters["sampling"]
    anchors, samples = sample_anchors(dynamic, sampling_parameters, work_path)

    transition_manifold = approximate_tm(dynamic, samples, work_path)

    linear_regression(simulation_parameters, transition_manifold, anchors, dynamic, work_path)

    return
