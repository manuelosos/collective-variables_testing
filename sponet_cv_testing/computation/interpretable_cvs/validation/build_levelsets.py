import networkx as nx
import numba
import numpy as np
from matplotlib import pyplot as plt


def create_x_from_levelset(num_x, partition, degrees, share_of_ones):
    """
    Create states x that belong to the same levelset of the CV.

    (The CV is assumed to be of the form as in "optimize_cv.py", so that states x
    with equal shares in each community will have the same CV value.)

    Parameters
    ----------
    num_x : int
        number of states x that are created
    partition : np.ndarray
        Node i is assigned to community partition[i]. The partitions are numbered (0,1,...),
        shape = (num_nodes,).
    degrees : np.ndarray
        degree[i] is the degree of node i.
    share_of_ones : np.ndarray
        Degree-weighted share of ones in each of the communities, each entry is normalized to [0,1],
        shape = (num_communities,).

    Returns
    -------
    np.ndarray
        shape = (num_x, num_nodes).
    """

    num_nodes = len(partition)
    num_communities = int(np.max(partition) + 1)
    mask_communities = [partition == i for i in range(num_communities)]
    idx_communities = [
        np.arange(num_nodes)[mask_communities[i]] for i in range(num_communities)
    ]

    out = np.zeros((num_x, num_nodes), dtype=int)

    for i in range(num_x):
        for j in range(num_communities):
            x_this_community = np.zeros(len(idx_communities[j]), dtype=int)
            x_this_community[: int(x_this_community.shape[0] * share_of_ones[j])] = 1
            np.random.shuffle(x_this_community)

            degrees_this_community = degrees[mask_communities[j]]
            this_share = np.inner(x_this_community, degrees_this_community)
            this_share_target = share_of_ones[j] * np.sum(degrees_this_community)

            while abs(this_share - this_share_target) >= np.min(degrees_this_community):
                if this_share > this_share_target:  # remove random "one"
                    idx_ones = np.nonzero(x_this_community == 1)[0]
                    idx_remove = np.random.choice(idx_ones)
                    x_this_community[idx_remove] = 0
                    this_share -= degrees_this_community[idx_remove]
                else:  # add random "one"
                    idx_zeros = np.nonzero(x_this_community == 0)[0]
                    idx_add = np.random.choice(idx_zeros)
                    x_this_community[idx_add] = 1
                    this_share += degrees_this_community[idx_add]

            out[i, mask_communities[j]] = x_this_community

    return out


def sample_state_like_x(
    x: np.ndarray,
    weights: np.ndarray,
    num_samples: int = 1,
    max_iter: int = 10**7,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Find states with identical CVs <x, weights> by random walk (MCMC).

    Parameters
    ----------
    x : np.ndarray
        shape = (num_agents,)
    weights : np.ndarray
        shape = (num_weights, num_agents)
    num_samples : int, optional
        number of similar states to sample
    max_iter : int, optional
        maximum number of iterations since last improvement
    tol : float, optional
        the random walk terminates if it found a y with ||CV(x)-CV(y)|| < tol

    Returns
    -------
    np.ndarray
        shape = (num_agents,) if num_samples = 1 else (num_samples, num_agents)
    """
    target_cv = np.dot(weights, x)
    out = np.zeros((num_samples, x.shape[0]))

    for i in range(num_samples):
        # initial guess
        initial_y = np.zeros_like(x)
        num_ones = int(np.sum(x))
        initial_y[:num_ones] = 1
        np.random.shuffle(initial_y)
        cv = np.dot(weights, initial_y)

        (
            best_y,
            total_count,
            iterations_since_last_improvement,
        ) = _numba_sample_state_target_cv(
            target_cv, initial_y, cv, weights.T, max_iter, tol
        )
        out[i] = np.copy(best_y)
        # print(total_count)
        # print(iterations_since_last_improvement)
        # cv = np.dot(weights, best_y)
        # print(np.sum((target_cv - cv) ** 2))

    # if num_samples == 1:
    #     return out[0]
    return out


def sample_state_target_cv(
    target_cv: np.ndarray,
    weights: np.ndarray,
    num_samples: int = 1,
    max_iter: int = 10**7,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Find states with target CV by random walk (MCMC).

    Parameters
    ----------
    target_cv : np.ndarray
        shape = (num_weights,)
    weights : np.ndarray
        shape = (num_weights, num_agents)
    num_samples : int, optional
        number of similar states to sample
    max_iter : int, optional
        maximum number of iterations since last improvement
    tol : float, optional
        the random walk terminates if it found a y with ||CV(x)-CV(y)|| < tol

    Returns
    -------
    np.ndarray
        shape = (num_agents,) if num_samples = 1 else (num_samples, num_agents)
    """
    num_nodes = weights.shape[1]
    out = np.zeros((num_samples, num_nodes))

    for i in range(num_samples):
        # initial guess
        initial_y = np.random.randint(0, 2, num_nodes)
        cv = np.dot(weights, initial_y)
        print(cv)

        (
            best_y,
            total_count,
            iterations_since_last_improvement,
        ) = _numba_sample_state_target_cv(
            target_cv, initial_y, cv, weights.T, max_iter, tol
        )
        out[i] = np.copy(best_y)
        print(total_count)
        print(iterations_since_last_improvement)
        cv = np.dot(weights, best_y)
        print(np.sum((target_cv - cv) ** 2))

    if num_samples == 1:
        return out[0]
    return out


@numba.njit()
def _numba_sample_state_target_cv(target_cv, initial_y, cv, weights, max_iter, tol):
    """
    Find y with || <y, weights> - target_cv || < tol using MCMC.

    Parameters
    ----------
    target_cv : np.ndarray
    initial_y : np.ndarray
    cv : np.ndarray
    weights : np.ndarray
    max_iter : int
    tol : float

    Returns
    -------
    tuple[np.ndarray, int, int]
    """
    temp = np.sqrt(np.sum(weights**2, axis=1))
    temp = -np.mean(temp) / np.log(0.1)
    diff = np.sqrt(np.sum((target_cv - cv) ** 2))

    best_y = np.copy(initial_y)
    best_diff = diff

    iterations_since_last_improvement = 0
    total_count = 0

    while diff > tol and iterations_since_last_improvement <= max_iter:
        total_count += 1
        iterations_since_last_improvement += 1

        if total_count % 2 == 0:  # pick idx with state 1
            possible_idx = np.argwhere(initial_y == 1)[:, 0]
        else:  # pick idx with state 0
            possible_idx = np.argwhere(initial_y == 0)[:, 0]

        if len(possible_idx) == 0:
            continue
        rand_idx = np.random.choice(possible_idx)

        new_minus_old_state = 1 if initial_y[rand_idx] == 0 else -1
        new_cv = cv + new_minus_old_state * weights[rand_idx]
        new_diff = np.sqrt(np.sum((target_cv - new_cv) ** 2))

        probability_switch = 1 if new_diff < diff else np.exp((-new_diff + diff) / temp)
        switch = np.random.random() <= probability_switch

        if switch:
            diff = new_diff
            cv = new_cv
            initial_y[rand_idx] = 1 - initial_y[rand_idx]
            # print(np.sum(initial_y))

            if diff < best_diff:
                best_diff = diff
                best_y = np.copy(initial_y)
                iterations_since_last_improvement = 0

    return best_y, total_count, iterations_since_last_improvement


def plot_x_levelset(
    x: np.ndarray, network: nx.Graph, other_idx: int = None, layout="spring"
):
    n_cols = 1
    n_rows = min(x.shape[0], 3)
    if layout == "kamada kawai":
        pos = nx.kamada_kawai_layout(network)
    else:
        pos = nx.spring_layout(network, seed=100, k=0.09)

    if other_idx is None:
        other_idx = -1

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2, 5))
    for j in range(n_rows):
        img = nx.draw_networkx_nodes(
            network,
            pos=pos,
            ax=axs[j],
            node_color=x[j],
            node_size=15,
            cmap="coolwarm",
            alpha=1,
        )
        nx.draw_networkx_edges(network, pos, ax=axs[j], width=0.5)
        if j == other_idx:
            axs[j].set_ylabel(f"y0")
        elif j < other_idx:
            axs[j].set_ylabel(f"$x^{j + 1}$", rotation=0, labelpad=10)
        else:
            axs[j].set_ylabel(f"$x^{j + 1}$", rotation=0, labelpad=10)

    return fig
