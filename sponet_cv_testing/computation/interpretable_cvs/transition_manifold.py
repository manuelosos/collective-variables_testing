import numpy as np
from numba import njit, prange
import scipy.sparse.linalg as sla
import logging
import numba

logger = logging.getLogger("sponet_cv_testing.computation")

class TransitionManifold:
    def __init__(
        self,
        bandwidth_transitions: float,
        bandwidth_diffusion_map: float = 1,
        dimension: int = 10,
    ):
        self.bandwidth_transitions = bandwidth_transitions
        self.bandwidth_diffusion_map = bandwidth_diffusion_map
        self.num_coordinates = dimension
        self.eigenvalues = None
        self.eigenvectors = None
        self.distance_matrix = None
        self.dimension_estimate = None

    def fit(self, x_samples: np.ndarray, optimize_bandwidth: bool = False):
        """
        Parameters
        ----------
        x_samples : np.ndarray
            Array containing the endpoints of num_samples simulations for each anchor point.
            Shape = (num_anchor_points, num_samples, dimension).
        optimize_bandwidth : bool, optional
            If true, the diffusion_bandwidth is optimized.
            This also yields an estimation of the transition manifold dimension, self.dimension_estimate.
        Returns
        -------
        np.ndarray
            Array containing the coordinates of each anchor point in diffusion space.
            Shape = (num_anchor_points, dimension).
        """
        logger.debug(f"Generating distance matrix")
        self.set_distance_matrix(x_samples)
        if optimize_bandwidth:
            logger.debug("Optimizing bandwidth of diffusion map")
            self.optimize_bandwidth_diffusion_maps()

        logger.debug("Calculating diffusion maps")
        return self.calc_diffusion_map()

    def set_distance_matrix(self, x_samples: np.ndarray):
        self.distance_matrix, _ = _numba_dist_matrix_gaussian_kernel(
            x_samples, self.bandwidth_transitions
        )

    def optimize_bandwidth_diffusion_maps(self):
        """
        Optimize and set diffusion bandwidth by calculating
        d log(S(e)) / d log(e)
        where S(e) is the average of the kernel matrix K = exp(-D^2 / e).

        Returns
        -------
        tuple(np.ndarray, np.ndarray, np.ndarray)
            e, S(e), d log(S(e)) / d log(e)
        """
        if self.distance_matrix is None:
            raise RuntimeError("No distance matrix available. Call the set_distance_matrix method first!")

        epsilons = np.logspace(-6, 2, 101)
        s = []
        for epsilon in epsilons:
            s.append(self.average_kernel_matrix(epsilon))
        s = np.array(s)

        derivative = _central_differences(np.log(epsilons), np.log(s))
        optim_idx = np.argmax(derivative)
        optim_epsilon = epsilons[optim_idx]
        self.bandwidth_diffusion_map = optim_epsilon ** 0.5
        self.dimension_estimate = derivative[optim_idx]

        return epsilons, s, derivative

    def calc_diffusion_map(self):
        if self.distance_matrix is None:
            raise RuntimeError("No distance matrix available. Call the set_distance_matrix method first!")

        self.eigenvalues, self.eigenvectors = calc_diffusion_maps(
            self.distance_matrix, self.num_coordinates, self.bandwidth_diffusion_map
        )
        return self.eigenvectors.real[:, 1:] * self.eigenvalues.real[np.newaxis, 1:]

    def average_kernel_matrix(self, epsilon) -> float:
        """
        epsilon = (bandwidth_diffusion_map)^2
        """
        if self.distance_matrix is None:
            raise RuntimeError("No distance matrix available. Call the set_distance_matrix method first!")

        return np.mean(np.exp(-(self.distance_matrix ** 2) / epsilon))


@njit(parallel=True)
def _numba_dist_matrix_gaussian_kernel(
    x_samples: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    x_samples : np.ndarray
        Shape = (num_anchor_points, num_samples, dimension).
    sigma : float
        Bandwidth.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        1) Distance matrix, shape = (num_anchor_points, num_anchor_points).
        2) kernel matrix diagonal, shape = (num_anchor_points,)
    """
    num_anchor, num_samples, dimension = x_samples.shape

    # compute symmetric kernel evaluations
    kernel_diagonal = np.zeros(num_anchor)
    for i in range(num_anchor):
        kernel_diagonal[i] = _numba_gaussian_kernel_eval(
            x_samples[i], x_samples[i], sigma
        )
    # compute asymmetric kernel evaluations and assemble distance matrix
    distance_matrix = np.zeros((num_anchor, num_anchor))
    for i in prange(num_anchor):
        for j in range(i):
            this_sum = _numba_gaussian_kernel_eval(x_samples[i], x_samples[j], sigma)
            distance_matrix[i, j] = (
                kernel_diagonal[i] + kernel_diagonal[j] - 2 * this_sum
            )
    distance_matrix /= num_samples**2
    return distance_matrix + np.transpose(distance_matrix), kernel_diagonal


@njit(parallel=False)
def _numba_gaussian_kernel_eval(x: np.ndarray, y: np.ndarray, sigma: float):
    """
    Parameters
    ----------
    x : np.ndarray
        shape = (# x points, dimension)
    y : np.ndarray
        shape = (# y points, dimension)
    sigma : float
        bandwidth

    Returns
    -------
    float
        sum of kernel matrix
    """
    nx = x.shape[0]
    ny = y.shape[0]

    X = np.sum(x * x, axis=1).reshape((nx, 1)) * np.ones((1, ny))
    Y = np.sum(y * y, axis=1) * np.ones((nx, 1))
    out = X + Y - 2 * np.dot(x, y.T)
    out /= -(sigma**2)
    np.exp(out, out)
    return np.sum(out)


def calc_diffusion_maps(
    distance_matrix: np.ndarray, num_components: int, sigma: float, alpha: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve diffusion map eigenproblem

    Parameters
    ----------
    distance_matrix : np.ndarray
        Symmetric array of shape (n_features, n_features).
    num_components : int
        Number of eigenvalues and eigenvectors to compute.
    sigma : float
        Diffusion map kernel bandwidth parameter.
    alpha : float
        Diffusion map normalization parameter.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (eigenvalues, eigenvectors).
    """

    num_points = distance_matrix.shape[0]
    kernel_matrix = np.exp(-(distance_matrix**2) / sigma**2)
    row_sum = np.sum(kernel_matrix, axis=0)

    # compensating for testpoint density
    kernel_matrix = kernel_matrix / np.outer(row_sum**alpha, row_sum**alpha)

    # row normalization
    kernel_matrix = kernel_matrix / np.tile(sum(kernel_matrix, 0), (num_points, 1))

    # weight matrix
    weight_matrix = np.diag(sum(kernel_matrix, 0))

    # solve the diffusion maps eigenproblem
    return sla.eigs(kernel_matrix, num_components + 1, weight_matrix)


def _central_differences(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute dy/dx via central differences.
    """
    out = np.zeros(len(x))
    for i in range(len(x)):
        upper_idx = min(i + 1, len(x) - 1)
        lower_idx = max(i - 1, 0)
        out[i] = (y[upper_idx] - y[lower_idx]) / (x[upper_idx] - x[lower_idx])
    return out
