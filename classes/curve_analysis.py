import numpy as np
from typing import List, Tuple, Dict, Set
from scipy.stats import wasserstein_distance

class CurveAnalysis:
    """Class for analyzing and comparing curves."""

    @staticmethod
    def normalize_curves(times: List[float], deduplicated_counts: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize the curves."""
        times = np.array(times)
        deduplicated_counts = np.array(deduplicated_counts)
        sort_indices = np.argsort(times)
        times = times[sort_indices]
        deduplicated_counts = deduplicated_counts[sort_indices]
        min_count, max_count = np.min(deduplicated_counts), np.max(deduplicated_counts)
        if max_count > min_count:
            normalized_counts = (deduplicated_counts - min_count) / (max_count - min_count)
        else:
            normalized_counts = np.ones_like(deduplicated_counts)
        return times, normalized_counts

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Kullback-Leibler divergence."""
        p = np.asarray(p)
        q = np.asarray(q)
        p = p / np.sum(p)
        q = q / np.sum(q)

        with np.errstate(divide='ignore', invalid='ignore'):
            div = np.where((p != 0) & (q != 0), p * np.log(p / q), 0)

        div[p != 0] = np.where(q[p != 0] == 0, np.inf, div[p != 0])

        return np.sum(div)

    @staticmethod
    def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence."""
        p = np.asarray(p)
        q = np.asarray(q)
        m = 0.5 * (p + q)
        return 0.5 * (CurveAnalysis.kl_divergence(p, m) + CurveAnalysis.kl_divergence(q, m))

    @staticmethod
    def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Jensen-Shannon distance."""
        return np.sqrt(CurveAnalysis.jensen_shannon_divergence(p, q))

    @staticmethod
    def earth_movers_distance(p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Earth Mover's distance."""
        return wasserstein_distance(p, q)

    @staticmethod
    def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Bhattacharyya distance."""
        return -np.log(np.sum(np.sqrt(p * q)))

    @staticmethod
    def area_between_curves(times: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate area between curves."""
        return simpson(y=np.abs(p - q), x=times)

    @staticmethod
    def chi_square_distance(p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Chi-Square distance."""
        return np.sum((p - q)**2 / (p + q))

    @staticmethod
    def kolmogorov_smirnov_test(p: np.ndarray, q: np.ndarray) -> float:
        """Perform Kolmogorov-Smirnov test."""
        return ks_2samp(p, q).statistic

    @staticmethod
    def correlation_coefficient(p: np.ndarray, q: np.ndarray) -> float:
        """Calculate correlation coefficient."""
        return np.corrcoef(p, q)[0, 1]

    @staticmethod
    def euclidean_distance(p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Euclidean distance."""
        return np.sqrt(np.sum((p - q)**2))

    @staticmethod
    def safe_log2(x: np.ndarray) -> np.ndarray:
        """Compute safe log2."""
        return np.log2(np.maximum(x, np.finfo(float).tiny))

    @staticmethod
    def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
        """Calculate cross entropy."""
        p, q = np.asarray(p), np.asarray(q)
        p, q = p / np.sum(p), q / np.sum(q)
        return -np.sum(p * CurveAnalysis.safe_log2(q))

    @staticmethod
    def joint_entropy(p: np.ndarray, q: np.ndarray) -> float:
        """Calculate joint entropy."""
        p, q = np.asarray(p), np.asarray(q)
        p, q = p / np.sum(p), q / np.sum(q)
        joint_dist = np.outer(p, q)
        return -np.sum(joint_dist * CurveAnalysis.safe_log2(joint_dist))

    @staticmethod
    def conditional_entropy(p: np.ndarray, q: np.ndarray) -> float:
        """Calculate conditional entropy."""
        return CurveAnalysis.joint_entropy(p, q) - entropy(q)

    @staticmethod
    def information_gain(p: np.ndarray, q: np.ndarray) -> float:
        """Calculate information gain."""
        return entropy(p) - CurveAnalysis.conditional_entropy(p, q)

    @staticmethod
    def interpolate_data(times: np.ndarray, counts: np.ndarray, new_times: np.ndarray) -> np.ndarray:
        """Interpolate data."""
        f = interp1d(times, counts, kind='linear', bounds_error=False, fill_value='extrapolate')
        return f(new_times)
