import numpy as np
from sponet import CNTMParameters, sample_many_runs


def sample_cntm(
    x_anchor: np.ndarray,
    num_samples: int,
    lag_time: float,
    params: CNTMParameters,
    n_jobs: int = -1,
):
    """

    Parameters
    ----------
    x_anchor : np.ndarray
        shape = (num_anchor_points, num_nodes)
    num_samples : int
    lag_time : float
    params : Parameters
    n_jobs : int
        Number of cpu cores to use. Default: -1 (all cores).

    Returns
    -------
    np.ndarray
        shape = (num_anchor_points, num_samples, num_nodes)
    """
    if n_jobs in [0, 1]:
        n_jobs = None
    t, x_out = sample_many_runs(
        params, x_anchor, lag_time, 2, num_samples, n_jobs=n_jobs
    )
    x_out = x_out[:, :, -1, :]
    return x_out
