import numpy as np
from sponet.collective_variables import CollectiveVariable
from sponet import sample_many_runs, Parameters
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick


class MinorSymLogLocator(mtick.Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """

    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        "Return the locations of the ticks"
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i - 1]
            if abs(majorlocs[i - 1] + majorstep / 2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError(
            "Cannot get tick locations for a " "%s type." % type(self)
        )


def calc_mean_and_variance(
    x_init: np.ndarray,
    params: Parameters,
    lag_time: float,
    cv: CollectiveVariable,
    num_samples: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    x_init : np.ndarray
        shape=(num_initial_states, dim_x)
    params : Parameters
    lag_time : float
    cv : CollectiveVariable
    num_samples : int, optional

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        1. means, shape=(num_initial_states, dim_cv)
        2. variances, shape=(num_initial_states, dim_cv)

    """
    t, x_end = sample_many_runs(
        params,
        x_init,
        lag_time,
        2,
        num_runs=num_samples,
        collective_variable=cv,
        n_jobs=-1,
    )
    x_end = x_end[:, :, -1, :]
    means = np.mean(x_end, axis=1)
    variances = np.var(x_end, axis=1)
    return means, variances


def plot_mean_and_var(means, variances, other_idx=None):
    if other_idx is None:
        other_idx = -1

    avg_mean = np.mean(means[np.arange(means.shape[0]) != other_idx], axis=0)
    deviation_from_avg_mean = np.linalg.norm(means - avg_mean, axis=1) / np.linalg.norm(
        avg_mean
    )

    avg_var = np.mean(variances[np.arange(variances.shape[0]) != other_idx], axis=0)
    deviation_from_avg_var = np.linalg.norm(
        variances - avg_var, axis=1
    ) / np.linalg.norm(avg_var)

    x_labels = []
    for j in range(means.shape[0]):
        if j == other_idx:
            x_labels.append("y0")
        elif j < other_idx:
            x_labels.append(f"x{j}")
        else:
            x_labels.append(f"x{j - 1}")

    fig, axes = plt.subplots(2, sharex=True)
    linthresh = 0.01
    axes[0].bar(x_labels, deviation_from_avg_mean)
    axes[0].set_yscale("symlog", linthresh=linthresh)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    axes[0].yaxis.set_minor_locator(MinorSymLogLocator(linthresh))
    axes[0].set_ylabel("deviation of mean")
    axes[0].grid(axis="y")

    axes[1].bar(x_labels, deviation_from_avg_var)
    axes[1].set_yscale("symlog", linthresh=linthresh)
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    axes[1].yaxis.set_minor_locator(MinorSymLogLocator(linthresh))
    axes[1].set_ylabel("deviation of variance")
    axes[1].grid(axis="y")

    return fig


def calc_mean_and_std_traj(
    x_init: np.ndarray,
    params: Parameters,
    t_max: float,
    num_steps: int,
    cv: CollectiveVariable,
    num_samples: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    x_init : np.ndarray
        shape=(num_initial_states, dim_x)
    params : Parameters
    t_max : float
    num_steps : int
    cv : CollectiveVariable
    num_samples : int, optional

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        0. array of snapshot times
        1. mean of trajs, shape=(num_initial_states, num_steps, dim_cv)
        2. variances, shape=(num_initial_states, num_steps, dim_cv)

    """
    t, x_traj = sample_many_runs(
        params,
        x_init,
        t_max,
        num_steps,
        num_runs=num_samples,
        collective_variable=cv,
        n_jobs=-1,
    )
    mean_traj = np.mean(x_traj, axis=1)
    std_traj = np.std(x_traj, axis=1)

    return t, mean_traj, std_traj


def plot_trajs(t, mean_traj, std_traj, other_idx=None):
    n_rows = min(4, mean_traj.shape[2])
    num_x = min(3, mean_traj.shape[0])
    colors = ["blue", "green", "red"]

    if other_idx is None:
        other_idx = -1

    fig, axes = plt.subplots(n_rows, sharex=True)
    if n_rows == 1:
        axes = [axes]
    for i in range(n_rows):
        for j in range(num_x):
            if j == other_idx:
                label = "y0"
            elif j < other_idx:
                label = f"x{j}"
            else:
                label = f"x{j - 1}"
            axes[i].plot(t, mean_traj[j, :, i], label=label, c=colors[j])
            axes[i].fill_between(
                t,
                mean_traj[j, :, i] - std_traj[j, :, i],
                mean_traj[j, :, i] + std_traj[j, :, i],
                color=colors[j],
                alpha=0.3,
            )
        axes[i].legend()
        axes[i].set_ylabel(rf"$\xi_{i+1}$")
        axes[i].grid()
    axes[-1].set_xlabel("t")

    return fig
