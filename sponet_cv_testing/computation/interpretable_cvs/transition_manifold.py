import logging
import numpy as np
import scipy.sparse.linalg as sla
from numba import njit, prange
from scipy.sparse.linalg import ArpackNoConvergence, ArpackError
import ctypes

logger = logging.getLogger("testpipeline.compute.transition_manifold")

# Setup to access system time in numba compiled functions
# Only required in compute_distance_matrices()
get_system_clock = ctypes.pythonapi._PyTime_GetSystemClock
as_seconds_double = ctypes.pythonapi._PyTime_AsSecondsDouble
get_system_clock.argtypes = []
get_system_clock.restype = ctypes.c_int64
as_seconds_double.argtypes = [ctypes.c_int64]
as_seconds_double.restype = ctypes.c_double


@njit(parallel=False)
def _numba_get_time() -> float:
    system_clock = get_system_clock()
    current_time = as_seconds_double(system_clock)
    return current_time


def compute_diffusion_maps(distance_matrices: np.ndarray,
                           num_coordinates: int,
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the transition manifold of a series of distance_matrices.
    Diffusion maps will be computed for every timestep.
    The diffusion bandwidths will be optimized automatically.
    The diffusion maps are computed using the Arpack wrapper of scipy.
    If the eigenvalue problem does not converge and the optimized diffusion bandwidth is below a threshold depending on
    the maximal value of the corresponding distance matrix, the diffusion bandwidth will be set to the threshold.

    Parameters
    ----------
    distance_matrices : np.ndarray
        Array of distance matrices, shape = (num_time_steps, num_anchors, num_anchors).
    num_coordinates : int
        Number of coordinates of the diffusion maps that will be computed.

    Returns
    -------
    diffusion_maps, diffusion_maps_eigenvalues, dimension_estimates, bandwidth_diffusion_maps :
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        diffusion_maps.shape = (num_time_steps, num_initial_states, num_coordinates),
        diffusion_maps_eigenvalues.shape = (num_time_steps, num_coordinates+1),
        dimension_estimates.shape = (num_time_steps),
        bandwidth_diffusion_maps.shape = (num_time_steps)
    """

    num_time_steps = distance_matrices.shape[0]
    num_initial_states = distance_matrices.shape[1]

    diffusion_maps = np.empty((num_time_steps, num_initial_states, num_coordinates))
    diffusion_maps_eigenvalues = np.empty((num_time_steps, num_coordinates + 1))
    dimension_estimates = np.empty(num_time_steps)
    bandwidth_diffusion_maps = np.empty(num_time_steps)

    for i in range(num_time_steps):
        bandwidth_diffusion_maps[i], dimension_estimates[i] = (
            optimize_bandwidth_diffusion_map(distance_matrices[i, :, :]))
        try:
            eigenvalues, eigenvectors = calc_diffusion_maps(distance_matrices[i, :, :],
                                                            num_coordinates,
                                                            bandwidth_diffusion_maps[i])
        except (ArpackError, ArpackNoConvergence) as e:
            logger.debug("Arpackerror occurred. Checking if bandwidth is below threshold")

            bandwidth_tolerance: float = np.max(distance_matrices[i, :, :]) / np.sqrt(20)
            if bandwidth_diffusion_maps[i] < bandwidth_tolerance:
                logger.debug(
                    f"#slb Sigma {bandwidth_diffusion_maps[i]} smaller than bandwidth tolerance: {bandwidth_tolerance}. "
                    f"Set sigma to {bandwidth_tolerance}")
                eigenvalues, eigenvectors = calc_diffusion_maps(distance_matrices[i, :, :], num_coordinates,
                                                                bandwidth_tolerance)

            else:
                logger.error(f"Bandwidth not below threshold {bandwidth_tolerance}")
                raise e

        diffusion_maps_eigenvalues[i, :] = eigenvalues
        diffusion_maps[i, :, :] = eigenvectors.real[:, 1:] * eigenvalues.real[np.newaxis, 1:]

    return diffusion_maps, diffusion_maps_eigenvalues, dimension_estimates, bandwidth_diffusion_maps


@njit(parallel=True)
def compute_distance_matrices(samples: np.ndarray,
                              bandwidth_transition: float,
                              distance_matrix_triangle_inequality_speedup: bool,
                              ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Computes the distance matrices for a set of samples.
    If distance_matrix_triangle_inequality_speedup is set to True, the distance matrices are computed in parallel.
    Otherwise, each distance matrix is computed parallely.

    Parameters
    ----------
    samples : np.ndarray
        Array of samples, shape = (num_initial_states, num_runs, num_time_steps, num_agents).
    bandwidth_transition : float
        Bandwidth sigma of transition kernel for the distance matrices.
    distance_matrix_triangle_inequality_speedup : bool
        If set to True, the distance matrices will be computed lazily utilizing estimates from the triangle inequality.
        This can lead to inexact results.

    Returns
    -------
    distance_matrices : np.ndarray | (distance_matrices, compute_times) : tuple[np.ndarray, np.ndarray]
        distance_matrices.shape = (num_time_steps, num_initial_states, num_initial_states)
        compute_times.shape = num_timesteps
    """

    num_time_steps = samples.shape[2]
    num_initial_states = samples.shape[0]

    distance_matrices = np.empty((num_time_steps, num_initial_states, num_initial_states))
    compute_times = np.empty(num_time_steps)

    if distance_matrix_triangle_inequality_speedup:
        for i in prange(num_time_steps):
            start_time = _numba_get_time()
            distance_matrices[i, :, :], _ = _numba_dist_matrix_gaussian_kernel_triangle_speedup(samples[:, :, i, :],
                                                                                                bandwidth_transition)
            compute_times[i] = _numba_get_time() - start_time
    else:

        for i in range(num_time_steps):
            start_time = _numba_get_time()
            distance_matrices[i, :, :], _ = _numba_dist_matrix_gaussian_kernel(samples[:, :, i, :],
                                                                               bandwidth_transition)
            compute_times[i] = _numba_get_time() - start_time

    return distance_matrices, compute_times


def optimize_bandwidth_diffusion_map(distance_matrix: np.ndarray) -> tuple[float, float]:
    """
    Optimizes the diffusion bandwidth and computes an estimate for the intrinsic dimension of the sample corresponding
    to the distance matrix.

    Parameters
    ----------
        distance_matrix : np.ndarray
    Returns
    -------
        optim_bandwidth_diffusion_map, dimension_estimate : tuple[float, float]

    """
    epsilons = np.logspace(-6, 2, 101)

    s = []
    for epsilon in epsilons:
        s.append(np.mean(np.exp(-(distance_matrix ** 2) / epsilon)))
    s = np.array(s)

    derivative = _central_differences(np.log(epsilons), np.log(s))
    optim_idx = np.argmax(derivative)
    optim_epsilon = epsilons[optim_idx]
    optim_bandwidth_diffusion_map: float = optim_epsilon ** 0.5
    dimension_estimate: float = derivative[optim_idx]

    return optim_bandwidth_diffusion_map, dimension_estimate


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
    distance_matrix /= num_samples ** 2
    return distance_matrix + np.transpose(distance_matrix), kernel_diagonal


@njit(parallel=False)
def _numba_dist_matrix_gaussian_kernel_triangle_speedup(
        x_samples: np.ndarray, sigma: float, triangle_speedup_tolerance=1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """
        Parameters
        ----------
        x_samples : np.ndarray
            Shape = (num_anchor_points, num_samples, dimension).
        sigma : float
            Bandwidth.
        triangle_speedup_tolerance : float
            Tolerance for relative difference between lower and upper triangle inequality bound for applying the
            speedup.
            Defaults to 10e-4.

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

    n_speedup = 0

    distance_matrix = np.zeros((num_anchor, num_anchor))
    for i in range(num_anchor):
        for j in range(i):

            # Check if triangle inequality speedup can be applied
            lower_max = -np.inf
            upper_min = np.inf
            for k in range(j):
                lower_triang_bound = np.abs(distance_matrix[i, k] - distance_matrix[j, k])
                upper_triang_bound = distance_matrix[i, k] + distance_matrix[j, k]

                if lower_triang_bound > lower_max:
                    lower_max = lower_triang_bound
                if upper_triang_bound < upper_min:
                    upper_min = upper_triang_bound

            # If difference is within tolerance, replace exact computation with mean of upper- and lower bound
            if (upper_min - lower_max) / lower_max < triangle_speedup_tolerance:
                distance_matrix[i, j] = 0.5 * (lower_max + upper_min)
                n_speedup = n_speedup + 1
                continue

            this_sum = _numba_gaussian_kernel_eval(x_samples[i], x_samples[j], sigma)
            distance_matrix[i, j] = (
                    kernel_diagonal[i] + kernel_diagonal[j] - 2 * this_sum
            )

    distance_matrix /= num_samples ** 2
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
    out /= -(sigma ** 2)
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
    kernel_matrix = np.exp(-(distance_matrix ** 2) / sigma ** 2)
    row_sum = np.sum(kernel_matrix, axis=0)

    # compensating for testpoint density
    kernel_matrix = kernel_matrix / np.outer(row_sum ** alpha, row_sum ** alpha)

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


# Deprecated
def compute_transition_manifold(samples: np.ndarray,
                                bandwidth_transitions: float,
                                num_coordinates: int,
                                distance_matrix_triangle_inequality_speedup: bool = False,
                                optimize_diffusion_map_bandwidth: bool = True,
                                bandwidth_diffusion_map: float | None = None
                                ) -> tuple[np.ndarray, np.ndarray, float | None, float]:
    if distance_matrix_triangle_inequality_speedup:
        distance_matrix, _ = _numba_dist_matrix_gaussian_kernel_triangle_speedup(samples, bandwidth_transitions)
    else:
        distance_matrix, _ = _numba_dist_matrix_gaussian_kernel(samples, bandwidth_transitions)

    dimension_estimate = None
    if optimize_diffusion_map_bandwidth:
        bandwidth_diffusion_map, dimension_estimate = optimize_bandwidth_diffusion_map(distance_matrix)
    assert bandwidth_diffusion_map is not None

    try:
        eigenvalues, eigenvectors = calc_diffusion_maps(distance_matrix, num_coordinates, bandwidth_diffusion_map)
    except (ArpackError, ArpackNoConvergence) as e:
        logger.debug("Arpackerror occurred. Checking if bandwidth is below threshold")

        bandwidth_tolerance: float = np.max(distance_matrix) / np.sqrt(20)
        if bandwidth_diffusion_map < bandwidth_tolerance:
            logger.debug(
                f"#slb Sigma {bandwidth_diffusion_map} smaller than bandwidth tolerance: {bandwidth_tolerance}. "
                f"Set sigma to {bandwidth_tolerance}")
            eigenvalues, eigenvectors = calc_diffusion_maps(distance_matrix, num_coordinates, bandwidth_tolerance)

        else:
            logger.error(f"Bandwidth not below threshold {bandwidth_tolerance}")
            raise e

    diffusion_map = eigenvectors.real[:, 1:] * eigenvalues.real[np.newaxis, 1:]

    return diffusion_map, eigenvalues, dimension_estimate, bandwidth_diffusion_map


# Deprecated
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
        self.distance_matrix_triangle_speedup = False

    def fit(self, x_samples: np.ndarray, optimize_bandwidth: bool = False, triangle_speedup: bool = False):
        """
        Parameters
        ----------
        x_samples : np.ndarray
            Array containing the endpoints of num_samples simulations for each anchor point.
            Shape = (num_anchor_points, num_samples, dimension).
        optimize_bandwidth : bool, optional
            If true, the diffusion_bandwidth is optimized.
            This also yields an estimation of the transition manifold dimension, self.dimension_estimate.
        triangle_speedup : bool, optional
        Returns
        -------
        np.ndarray
            Array containing the coordinates of each anchor point in diffusion space.
            Shape = (num_anchor_points, dimension).
        """
        self.distance_matrix_triangle_speedup = triangle_speedup

        if self.distance_matrix is None:
            logger.debug(f"Generating distance matrix")
            self.set_distance_matrix(x_samples)
        if optimize_bandwidth:
            logger.debug("Optimizing bandwidth of diffusion map")
            self.optimize_bandwidth_diffusion_maps()

        logger.debug("Calculating diffusion maps")
        diffusion_map = self.calc_diffusion_map()

        return diffusion_map

    def set_distance_matrix(self, x_samples: np.ndarray):

        if self.distance_matrix_triangle_speedup:
            logger.debug("Triangle Inequality speedup enabled")
            self.distance_matrix, _ = _numba_dist_matrix_gaussian_kernel_triangle_speedup(
                x_samples, self.bandwidth_transitions
            )
        else:
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

        logger.debug(f"Optimal bandwidth: {self.bandwidth_diffusion_map}")

        return epsilons, s, derivative

    def calc_diffusion_map(self):
        if self.distance_matrix is None:
            raise RuntimeError("No distance matrix available. Call the set_distance_matrix method first!")

        try:
            self.eigenvalues, self.eigenvectors = calc_diffusion_maps(
                self.distance_matrix, self.num_coordinates, self.bandwidth_diffusion_map
            )
        except (ArpackError, ArpackNoConvergence) as e:

            logger.debug("Arpackerror occurred. Checking if bandwidth is below threshold")

            bandwidth_tolerance: float = np.max(self.distance_matrix) / np.sqrt(20)

            if self.bandwidth_diffusion_map < bandwidth_tolerance:
                logger.debug(
                    f"#slb Sigma {self.bandwidth_diffusion_map} smaller than bandwidth tolerance: {bandwidth_tolerance}. "
                    f"Set sigma to {bandwidth_tolerance}")
                self.eigenvalues, self.eigenvectors = calc_diffusion_maps(
                    self.distance_matrix, self.num_coordinates, bandwidth_tolerance
                )
            else:
                logger.error("Bandwidth not below threshold.")
                raise e

        return self.eigenvectors.real[:, 1:] * self.eigenvalues.real[np.newaxis, 1:]

    def average_kernel_matrix(self, epsilon) -> float:
        """
        epsilon = (bandwidth_diffusion_map)^2
        """
        if self.distance_matrix is None:
            raise RuntimeError("No distance matrix available. Call the set_distance_matrix method first!")

        return np.mean(np.exp(-(self.distance_matrix ** 2) / epsilon))



