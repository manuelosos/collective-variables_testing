import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


def calc_mmd(x: np.ndarray, sigma: float = None):
    """
    Calculate the pairwise maximum mean discrepancy for different data sets.

    The MMD is calculated w.r.t. the gaussian kernel with bandwidth sigma.

    Parameters
    ----------
    x : np.ndarray
        shape = (num_random_variables, num_samples, num_features)
    sigma : float, optional
        bandwidth

    Returns
    -------
    np.ndarray
        shape = (num_random_variables, num_random_variables)
    """
    num_random_variables, num_samples, num_features = x.shape

    if sigma is None:
        gamma = None
    else:
        gamma = 1 / sigma**2

    kernel_mat = np.zeros((num_random_variables, num_random_variables))
    for i in range(num_random_variables):
        for j in range(min(i + 1, num_random_variables)):
            kernel_mat[i, j] = np.mean(
                pairwise_kernels(x[i], x[j], metric="rbf", gamma=gamma)
            )

    dist_mat = np.zeros((num_random_variables, num_random_variables))
    for i in range(num_random_variables):
        for j in range(i):
            dist_mat[i, j] = kernel_mat[i, i] + kernel_mat[j, j] - 2 * kernel_mat[i, j]

    dist_mat += dist_mat.T
    return dist_mat


def mmd_from_trajs(c):
    mmd = np.zeros((c.shape[2], c.shape[0], c.shape[0]))
    for i in range(c.shape[2]):
        #  print(i)
        mmd[i] = calc_mmd(c[:, :, i, :])

    return mmd


def plot_validation_mmd(t, mmd, ax, label, t_star: float = None):
    for i in range(mmd.shape[1]):
        for j in range(i):
            ax.semilogy(t, mmd[:, j, i], label=fr"{label}$(x^{j + 1},x^{i + 1})$")
    mmd_no_zeros = mmd[mmd != 0]
    if np.min(mmd_no_zeros) < 1e-8:
        low_lim = 1e-8
    else:
        low_lim = 0.9 * np.min(mmd_no_zeros)

    if t_star is not None:
        ax.set_xticks(list(ax.get_xticks(minor=True)) + [t_star], labels=list(ax.get_xticklabels(minor=True)) + [' $t^*$'], minor=True)
        ax.axvline(t_star, color='k', linestyle='--', linewidth=1)

    ax.set_ylim((low_lim, 1.1 * np.max(mmd)))
    ax.grid(which="major")
    ax.legend(loc="upper right")
    ax.set_xlabel("t")
