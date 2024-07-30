import networkx as nx
import numpy as np
from scipy.stats import dirichlet
from sponet import Parameters, sample_many_runs


def create_random_anchor_points(
    num_agents: int, num_opinions: int, num_anchor_points: int
) -> np.ndarray:
    """
    Create approximately equally many anchor points per opinion count
    (uniform distribution on the simplex of opinion shares).
    Each anchor point is randomly shuffled.

    Parameters
    ----------
    num_agents : int
    num_opinions : int
    num_anchor_points : int

    Returns
    -------
    np.ndarray
    """
    x_anchor = np.zeros((num_anchor_points, num_agents))
    alpha = np.ones(num_opinions)
    for i in range(num_anchor_points):
        shares = dirichlet.rvs(alpha=alpha)[0]
        counts = np.round(shares * num_agents).astype(int)
        counts[-1] = num_agents - np.sum(counts[:-1])

        this_x = []
        for m in range(num_opinions):
            this_x += [m] * counts[m]
        np.random.shuffle(this_x)
        x_anchor[i] = this_x

    x_anchor = np.unique(x_anchor.astype(int), axis=0)

    while x_anchor.shape[0] != num_anchor_points:
        missing_points = num_anchor_points - x_anchor.shape[0]
        x_anchor = np.concatenate(
            [
                x_anchor,
                create_random_anchor_points(num_agents, num_opinions, missing_points),
            ]
        )
        x_anchor = np.unique(x_anchor.astype(int), axis=0)

    return x_anchor


def create_anchor_points_degree_weighted(
    network: nx.Graph, num_samples: int
) -> np.ndarray:
    """
    Create random anchor points with uniformly distributed weighted opinion shares.

    Only for M=2 opinions.
    The weights are the node degrees.
    For each sample, the weight is distributed differently between higher degree and lower degree nodes.

    Parameters
    ----------
    network : nx.Graph
    num_samples : int

    Returns
    -------
    np.ndarray
    """
    num_agents = network.number_of_nodes()
    x_anchor = []

    degree_sequence = np.array([d for n, d in network.degree()])
    prob_distr_low = (1 / degree_sequence) / np.sum(1 / degree_sequence)
    prob_distr_high = degree_sequence / np.sum(degree_sequence)
    node_sequence = np.array([n for n, d in network.degree()])
    total_weight = np.sum(degree_sequence)

    for i in range(num_samples):
        this_x = np.zeros(num_agents)

        weight = int(np.round(np.random.random() * total_weight))
        weight_low_degrees = int(np.round(np.random.random() * weight))
        weight_high_degrees = weight - weight_low_degrees

        nodes_low_degrees = np.random.choice(
            num_agents, num_agents, replace=False, p=prob_distr_low
        )
        this_weight = 0
        for i in range(num_agents):
            this_agent_idx = nodes_low_degrees[i]
            this_agent = node_sequence[this_agent_idx]
            this_weight += degree_sequence[this_agent_idx]
            if this_weight > weight_low_degrees:
                this_weight -= degree_sequence[this_agent_idx]
                continue
            this_x[this_agent] = 1
            if this_weight == weight_low_degrees:
                break

        nodes_high_degrees = np.random.choice(
            num_agents, num_agents, replace=False, p=prob_distr_high
        )
        this_weight = 0
        for i in range(num_agents):
            this_agent_idx = nodes_high_degrees[i]
            this_agent = node_sequence[this_agent_idx]
            if this_x[this_agent] == 1:
                continue
            this_weight += degree_sequence[this_agent_idx]
            if this_weight > weight_high_degrees:
                this_weight -= degree_sequence[this_agent_idx]
                continue
            this_x[this_agent] = 1
            if this_weight == weight_high_degrees:
                break
        x_anchor.append(this_x)

    x_anchor = np.array(x_anchor)
    x_anchor = np.unique(x_anchor, axis=0).astype(int)
    return x_anchor


def create_anchor_points_local_clusters(
    network: nx.Graph, num_opinions: int, num_anchor_points: int, max_num_seeds: int = 1, min_num_seeds: int = 1
) -> np.ndarray:
    """
    Create anchor points by the following procedure:
    1) Pick uniformly random opinion counts
    2) Pick num_seeds random seeds on the graph for each opinion (uniformly between min_num_seeds and max_num_seeds)
    3) Propagate the opinions outward from each seed to neighboring nodes until the counts are reached

    Parameters
    ----------
    network : nx.Graph
    num_opinions : int
    num_anchor_points : int
    max_num_seeds : int
    min_num_seeds : int

    Returns
    -------
    np.ndarray
    """

    if num_opinions <= 256:
        network_data_type = np.uint8
    else:
        network_data_type = np.uint16

    num_agents = network.number_of_nodes()
    x_anchor = np.zeros((num_anchor_points, num_agents), dtype=network_data_type)

    alpha = np.ones(num_opinions)

    for i in range(num_anchor_points):
        # print(i)
        target_shares = dirichlet.rvs(alpha=alpha)[0]
        target_counts = np.round(target_shares * num_agents).astype(int)
        target_counts[-1] = num_agents - np.sum(target_counts[:-1])
        x = -1 * np.ones(num_agents)  # -1 stands for not yet specified
        counts = np.zeros(num_opinions)  # keep track of current counts for each opinion

        # pick initial seeds
        num_seeds = int(np.random.randint(min_num_seeds, max_num_seeds + 1))
        seeds = np.random.choice(
            num_agents, size=num_seeds * num_opinions, replace=False
        )
        np.random.shuffle(seeds)
        seeds = seeds.reshape(
            (num_opinions, num_seeds)
        )  # keep track of seeds of each opinion
        seeds = list(seeds)

        counts_reached = np.zeros(num_opinions).astype(bool)

        while True:
            # iterate through seeds and propagate opinions
            opinions = np.array(range(num_opinions))
            np.random.shuffle(opinions)
            for m in opinions:
                # if counts are reached, there is nothing to do
                if counts_reached[m]:
                    continue

                # if there are no seeds available, add a random new one
                if len(seeds[m]) == 0:
                    possible_idx = np.nonzero(x == -1)[0]
                    new_seed = np.random.choice(possible_idx)
                    seeds[m] = np.array([new_seed])

                new_seeds_m = []
                # set opinion of seeds to m
                for seed in seeds[m]:
                    if x[seed] != -1:
                        continue

                    if counts[m] < target_counts[m]:
                        x[seed] = m
                        counts[m] += 1

                        # add neighbors that are available as new seeds
                        neighbors = np.array([n for n in network.neighbors(seed)])
                        neighbors = neighbors[x[neighbors] == -1]
                        new_seeds_m += neighbors.tolist()

                    if counts[m] == target_counts[m]:  # counts have been reached
                        counts_reached[m] = True
                        break

                new_seeds_m = np.unique(new_seeds_m)
                np.random.shuffle(new_seeds_m)
                seeds[m] = new_seeds_m

            if np.all(counts_reached):
                break

        x_anchor[i] = x

    x_anchor = np.unique(x_anchor.astype(network_data_type), axis=0)

    while x_anchor.shape[0] != num_anchor_points:
        missing_points = num_anchor_points - x_anchor.shape[0]
        x_anchor = np.concatenate(
            [
                x_anchor,
                create_anchor_points_local_clusters(
                    network, num_opinions, missing_points, max_num_seeds
                ),
            ]
        )
        x_anchor = np.unique(x_anchor.astype(network_data_type), axis=0)

    return x_anchor


def integrate_anchor_points(
    x_anchor: np.ndarray,
        params: Parameters,
        tau: float
) -> np.ndarray:
    _, x = sample_many_runs(params, x_anchor, tau, 2, 1, n_jobs=-1)
    return x[:, 0, -1, :]


def perturbate_anchor_points(x_anchor: np.ndarray, num_opinions: int, epsilon: float):
    """
    Perturbate each x in x_anchor by randomly switching the state of a fraction of epsilon nodes.
    """
    x = np.copy(x_anchor)
    num_nodes_perturbate = int(epsilon * x.shape[1])
    for i in range(x.shape[0]):
        idx_perturbate = np.random.choice(
            x.shape[1], num_nodes_perturbate, replace=False
        )
        new_states = np.random.randint(0, num_opinions, num_nodes_perturbate)
        x[i, idx_perturbate] = new_states
    return x
