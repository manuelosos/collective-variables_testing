import numpy as np
from sponet import Parameters, sample_many_runs
from matplotlib import pyplot as plt


def plot_c_rand_func(t: np.ndarray, c: np.ndarray):
    half_size = int(c.shape[0] / 2)
    fig, axes = plt.subplots(2, half_size, figsize=(10, 6))

    for k in [0, 1]:
        for i in range(half_size):
            idx = k * half_size + i
            axes[k, i].set_title(f"$<c_{idx}, x>$")
            for j in range(c.shape[1]):
                axes[k, i].plot(t, c[idx, j, :], label=f"$x_{j}$")
            axes[k, i].legend()
            axes[k, i].grid()

    return fig


def validate_rand_func(
    num_random_functions: int,
    params: Parameters,
    x_init: np.ndarray,
    t_max: float,
    num_timesteps: int,
    num_samples: int,
):
    rng = np.random.default_rng(seed=1)
    random_functions = rng.standard_normal((num_random_functions, params.num_agents))

    t, x = sample_many_runs(
        params, x_init, t_max, num_timesteps, num_runs=num_samples, n_jobs=2
    )

    c = np.tensordot(random_functions, x, axes=[[1], [3]])
    c = np.mean(c, axis=2)
    return t, c
