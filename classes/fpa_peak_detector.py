import numpy as np
from typing import List, Dict
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


class FPAPeakDetector:
    @staticmethod
    def find_consensus_peak(chromatograms: List[Dict], known_peaks: Dict[str, float], compound_type: str, truncations: List[str]) -> Dict:
        if not chromatograms:
            return {'peak': None, 'time': None, 'count': None, 'deduplicated_counts': None}

        all_peaks = []
        for chromatogram in chromatograms:
            deduplicated_counts = np.array(chromatogram['deduplicated_counts'])
            times = np.array(chromatogram['times'])
            counts = np.array(chromatogram['counts'])

            if len(deduplicated_counts) == 0 or len(times) == 0:
                continue

            times, deduplicated_counts = FPAPeakDetector._smooth_chromatogram(times, deduplicated_counts)
            baseline_corrected_counts = FPAPeakDetector._improved_baseline_correction(deduplicated_counts)
            peaks_info = FPAPeakDetector._adaptive_multi_pass_peak_detection(times, baseline_corrected_counts)
            all_peaks.extend(peaks_info)

        if not all_peaks:
            return {'peak': None, 'time': None, 'count': None, 'deduplicated_counts': None}

        sorted_peaks = sorted(all_peaks, key=lambda x: x['score'], reverse=True)

        if compound_type == 'double':
            return FPAPeakDetector._select_best_double_truncation_peak(sorted_peaks, known_peaks)
        elif compound_type == 'single':
            return FPAPeakDetector._select_best_single_truncation_peak(sorted_peaks, known_peaks, truncations)
        else:  # Full compound
            return FPAPeakDetector._select_best_full_compound_peak(sorted_peaks, known_peaks, truncations)


    @staticmethod
    def _smooth_chromatogram(times, counts, window_length=5, polyorder=2):
        if len(counts) < window_length:
            window_length = len(counts) if len(counts) % 2 != 0 else len(counts) - 1
            polyorder = min(polyorder, window_length - 1)

        if window_length > 2:
            smoothed_counts = savgol_filter(counts, window_length, polyorder)
        else:
            smoothed_counts = counts

        return times, smoothed_counts

    @staticmethod
    def _improved_baseline_correction(y, lam=100, p=0.001, niter=10):
        L = len(y)

        if L < 3:
            return y

        if L < 10:
            return y - np.min(y)

        try:
            D = sparse.diags([1,-2,1], [0, -1, -2], shape=(L,L-2))
            w = np.ones(L)
            for i in range(niter):
                W = sparse.spdiags(w, 0, L, L)
                Z = W + lam * D.dot(D.transpose())
                z = spsolve(Z, w*y)
                w = p * (y > z) + (1-p) * (y < z)
            return y - z
        except ValueError:
            return y - np.min(y)

    @staticmethod
    def _adaptive_multi_pass_peak_detection(times, counts):
        peaks_info = []

        # Calculate dynamic thresholds
        noise_level = np.percentile(counts, 10)
        signal_level = np.percentile(counts, 90)
        dynamic_threshold = noise_level + 0.2 * (signal_level - noise_level)  # Increased threshold

        # First pass: detect major peaks
        major_peaks, _ = find_peaks(counts, height=dynamic_threshold, distance=10, prominence=0.6*(signal_level - noise_level))

        # Second pass: detect minor peaks
        minor_peaks, _ = find_peaks(counts, height=0.7*dynamic_threshold, distance=7, prominence=0.4*(signal_level - noise_level))

        all_peaks = np.unique(np.concatenate([major_peaks, minor_peaks]))

        for peak in all_peaks:
            score = FPAPeakDetector._calculate_peak_score(counts, peak, times[peak])
            peaks_info.append({
                'peak': peak,
                'time': times[peak],
                'count': counts[peak],
                'deduplicated_counts': counts[peak],
                'score': score
            })

        return peaks_info

    @staticmethod
    def _calculate_peak_score(counts, peak_index, peak_time):
        prominence = counts[peak_index] - np.min(counts[max(0, peak_index-10):min(len(counts), peak_index+11)])
        width = FPAPeakDetector._estimate_peak_width(counts, peak_index)
        area = prominence * width
        symmetry = FPAPeakDetector._calculate_peak_symmetry(counts, peak_index)
        return area * symmetry * (1 + 0.05 * peak_time / np.max(counts))  # Reduced time factor, added symmetry

    @staticmethod
    def _estimate_peak_width(counts, peak_index):
        left_index = peak_index
        right_index = peak_index
        half_height = (counts[peak_index] + np.min(counts)) / 2

        while left_index > 0 and counts[left_index] > half_height:
            left_index -= 1
        while right_index < len(counts) - 1 and counts[right_index] > half_height:
            right_index += 1

        return right_index - left_index

    @staticmethod
    def _calculate_peak_symmetry(counts, peak_index):
        left_half = counts[max(0, peak_index-5):peak_index]
        right_half = counts[peak_index+1:min(len(counts), peak_index+6)]

        if len(left_half) == 0 or len(right_half) == 0:
            return 1.0

        left_area = np.sum(left_half)
        right_area = np.sum(right_half)

        symmetry = min(left_area, right_area) / max(left_area, right_area)
        return symmetry

    @staticmethod
    def _select_best_double_truncation_peak(sorted_peaks, known_peaks):
        valid_peaks = [p for p in sorted_peaks if p['time'] > known_peaks.get('AgxNull', 0)]
        return max(valid_peaks, key=lambda x: x['score']) if valid_peaks else None

    @staticmethod
    def _select_best_single_truncation_peak(sorted_peaks, known_peaks, truncations):
        max_dt_time = max(known_peaks.get(dt, 0) for dt in truncations)
        valid_peaks = [p for p in sorted_peaks if p['time'] > max_dt_time]
        return max(valid_peaks, key=lambda x: x['score']) if valid_peaks else None

    @staticmethod
    def _select_best_full_compound_peak(sorted_peaks, known_peaks, truncations):
        max_truncation_time = max(known_peaks.get(t, 0) for t in truncations)
        valid_peaks = [p for p in sorted_peaks if p['time'] > max_truncation_time]
        return max(valid_peaks, key=lambda x: x['score']) if valid_peaks else None

    @staticmethod
    def process_compound_hierarchy(chromatograms, compound_hierarchy):
        results = {}

        # Process Double Truncations
        for dt in compound_hierarchy['double_truncations']:
            results[dt] = FPAPeakDetector.find_consensus_peak(
                chromatograms,
                known_peaks={'AgxNull': 0},
                compound_type='double',
                truncations=[]
            )

        # Process Single Truncations
        for st, dt_pair in compound_hierarchy['single_truncations'].items():
            results[st] = FPAPeakDetector.find_consensus_peak(
                chromatograms,
                known_peaks={dt: results[dt]['time'] for dt in dt_pair if results[dt]['time'] is not None},
                compound_type='single',
                truncations=dt_pair
            )

        # Process Full Compound
        fc = compound_hierarchy['full_compound']
        all_truncations = list(compound_hierarchy['double_truncations']) + list(compound_hierarchy['single_truncations'].keys())
        results[fc] = FPAPeakDetector.find_consensus_peak(
            chromatograms,
            known_peaks={t: results[t]['time'] for t in all_truncations if results[t]['time'] is not None},
            compound_type='full',
            truncations=all_truncations
        )

        return results
