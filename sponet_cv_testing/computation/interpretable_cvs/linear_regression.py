import cvxpy as cp
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sponet.collective_variables import OpinionShares, CompositeCollectiveVariable


class GeneralizedLasso(BaseEstimator, RegressorMixin):
    def __init__(self, penalty_value: float = 0, penalty_matrix: np.ndarray = None):
        """
        Solves the generalized lasso problem:
        min_w ||X @ w - y||_2 + penalty_value * ||penalty_matrix @ w||_1

        Parameters
        ----------
        penalty_value : float, optional
        penalty_matrix : np.ndarray, optional
        """
        self.penalty_value = penalty_value
        self.penalty_matrix = penalty_matrix

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Parameters
        ----------
        X : np.ndarray,
            Shape = (num_samples, num_features).
        y : np.ndarray
            Shape = (num_samples,).
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        pen_mat = (
            self.penalty_matrix
            if self.penalty_matrix is not None
            else np.eye(X.shape[1])
        )
        alpha = cp.Variable(X.shape[1])
        objective = cp.Minimize(
            cp.sum_squares(X @ alpha - y) / 2 / X.shape[0]
            + self.penalty_value * cp.norm(pen_mat @ alpha, 1)
        )
        prob = cp.Problem(objective)
        prob.solve(verbose=False)
        self.coef_ = alpha.value

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray):
        """
        Parameters
        ----------
        X : np.ndarray
            Shape = (num_samples, num_features).
        Returns
        -------
        np.ndarray
            Shape = (num_samples,).
        """
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")
        return X @ self.coef_


def build_incidence_matrix(edge_list, num_nodes=None, normalize=False):
    """
    Build incidence matrix of shape (num_edges, num_nodes).

    The i-th row has a -1 at index edge_list[i][0] and +1 at index edge_list[i][1].

    Parameters
    ----------
    edge_list : list
        List of edges in form of 2-tuples, e.g., edge_list[i] = (j, k).
    num_nodes : int, optional
        If not specified, the number of nodes is set by the maximum of edge_list.
    normalize : bool, optional
        If True, the matrix is divided by the num_edges.

    Returns
    -------
    np.ndarray
    """
    edge_list = np.array(edge_list)
    n_edges = edge_list.shape[0]
    if num_nodes is None:
        num_nodes = int(np.max(edge_list)) + 1

    # incidence_mat = np.zeros((n_edges, num_nodes))
    incidence_mat = lil_matrix((n_edges, num_nodes))
    for idx, edge in enumerate(edge_list):
        incidence_mat[idx, edge[0]] = -1
        incidence_mat[idx, edge[1]] = 1

    incidence_mat = csr_matrix(incidence_mat)

    if normalize:
        incidence_mat /= n_edges

    return incidence_mat


def optimize_fused_lasso(
    x: np.ndarray,
    xi_data: np.ndarray,
    network: nx.Graph,
    pen_vals: list[float],
    weights: np.ndarray = None,
    performance_threshold: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    x : np.ndarray
        shape = (num_testpoints, dim_x)
    xi_data : np.ndarray
        Collective variable evaluations, shape = (num_testpoints, dim_xi).
    network : nx.Graph
    pen_vals : list[float]
        List of penalty values.
    weights : np.ndarray, optional
        Pre-weighting factor, shape = (dim_x,).
    performance_threshold : float, optional
        Choose the largest penalty value so that the performance is still
        performance_threshold times the optimal performance.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        1. optimal alpha, shape = (dim_x, dim_xi).
        2. xi_fit, shape = (num_testpoints, dim_xi).
    """

    # print("Applying linear regression...")
    xi_data_centered = xi_data - np.mean(xi_data, axis=0)

    ell = xi_data.shape[0]
    if weights is None:
        weights = np.ones_like(x)
    x_data = x * weights
    one_matrix = np.eye(ell) - np.ones((ell, ell)) / ell
    x_data_centered = one_matrix @ x_data

    edge_list = [e for e in network.edges]
    inc_mat = build_incidence_matrix(edge_list, normalize=True)

    optimal_alpha = np.zeros((x.shape[1], xi_data.shape[1]))
    param_grid = {"penalty_value": pen_vals, "penalty_matrix": [inc_mat]}
    for i in range(xi_data.shape[1]):
        #print(f"Coordinate {i + 1}...")
        model = GeneralizedLasso()
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=ShuffleSplit(n_splits=3),
            refit=False,
        )
        grid_search.fit(x_data_centered, xi_data_centered[:, i])

        # find the biggest pen_val that has at least 95% of best performance
        mean_scores = grid_search.cv_results_["mean_test_score"]
        best_score = np.max(mean_scores)
        best_pen_val = 0

        for pen_val, score in zip(pen_vals, mean_scores):
            if score >= performance_threshold * best_score and pen_val > best_pen_val:
                best_pen_val = pen_val

        model = GeneralizedLasso(penalty_value=best_pen_val, penalty_matrix=inc_mat)
        model.fit(x_data_centered, xi_data_centered[:, i])
        #print(f"best choice for regularization penalty: {best_pen_val}")
        optimal_alpha[:, i] = model.coef_

    xi_fit = x_data @ optimal_alpha

    return optimal_alpha, xi_fit


def build_cv_from_alpha(
    alphas: np.ndarray,
    num_opinions: int,
    weights: np.ndarray = None,
):
    """

    Parameters
    ----------
    alphas : np.ndarray
        shape = (num_nodes, cv_dim).
    num_opinions : int
    weights : np.ndarray, optional
        degrees[i] is the degree of node i, shape = (num_nodes,).

    Returns
    -------
    CompositeCollectiveVariable
    """
    num_nodes = alphas.shape[0]
    xis = []

    if weights is None:
        weights = np.ones(num_nodes)

    for i in range(alphas.shape[1]):
        alphas_all_nodes = np.array([alphas[j, i] for j in range(num_nodes)])
        cv_weights = weights * alphas_all_nodes
        xis.append(
            OpinionShares(num_opinions, True, weights=cv_weights, idx_to_return=1)
        )

    xi = CompositeCollectiveVariable(xis)
    return xi


def create_plot_optimal_cv(
    xi, network, alphas, colors, partition_values=None, plot_xi_zeta=True
):
    n_rows = xi.shape[1]
    n_cols = 2 if plot_xi_zeta else 1
    pos = nx.kamada_kawai_layout(network)
    # pos = nx.spring_layout(network, seed=100)

    if partition_values is None:
        partition_values = np.arange(network.number_of_nodes())

    height = 7 if n_rows > 1 else 2
    width = 7 if plot_xi_zeta else 3
    figsize = (width, height)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)

    v_min = np.min(alphas)
    v_max = np.max(alphas)
    # v_min, v_max = -0.042, 0.042

    for i in range(n_rows):
        node_color = np.zeros(network.number_of_nodes())
        for j in range(node_color.shape[0]):
            node_color[j] = alphas[partition_values[j], i]
            # node_color[j] = 1 if node_color[j] >= 0 else -1

        if plot_xi_zeta:
            ax0 = axs[i, 0] if n_rows > 1 else axs[0]
            ax1 = axs[i, 1] if n_rows > 1 else axs[1]

            ax0.plot(xi[:, i], colors[:, i], "x")
            ax0.set_ylabel(rf"$\zeta_{i + 1}$")
            if i == n_rows - 1:
                ax0.set_xlabel(rf"$\xi_{i + 1}$")
            ax0.grid()
            ax0.xaxis.set_major_locator(plt.MaxNLocator(3))

        else:
            ax1 = axs[i] if n_rows > 1 else axs

        img = nx.draw_networkx_nodes(
            network,
            pos=pos,
            ax=ax1,
            node_color=node_color,
            node_size=20,
            vmin=v_min,
            vmax=v_max,
        )
        nx.draw_networkx_edges(network, pos, ax=ax1)
        fig.colorbar(img, ax=ax1)

        # ax1.set_xlabel(r"$\xi_1$")

    return fig
