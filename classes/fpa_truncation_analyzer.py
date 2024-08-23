import numpy as np
from typing import List, Tuple, Dict, Set
from scipy.stats import wasserstein_distance
from fpa_data_preprocessor import FPADataPreprocessor
from fpa_peak_detector import FPAPeakDetector

class FPATruncationAnalyzer:
    def __init__(self, data_preprocessor: 'FPADataPreprocessor', peak_detector: 'FPAPeakDetector'):
        self.data_preprocessor = data_preprocessor
        self.peak_detector = peak_detector
        self.known_peaks = {}
        self.n_1 = data_preprocessor.n_1
        self.full_compound_dict = data_preprocessor.full_compound_dict

    def analyze_double_truncations(self, building_blocks: List[str], measures: Dict) -> Dict[str, Dict]:
        results = {}
        for bb in building_blocks:
            permutations = [(bb, self.n_1, self.n_1), (self.n_1, bb, self.n_1), (self.n_1, self.n_1, bb)]
            chromatograms = [chrom for perm in permutations for _, _, chrom in self.full_compound_dict.get(perm, [])]
            if not chromatograms:
                print(f"No chromatograms found for double truncation {bb}")
                results[bb] = {'peak': None, 'time': None, 'count': None, 'deduplicated_counts': None}
                continue
            peak = self.peak_detector.find_consensus_peak(chromatograms, compound_type='double')
            if peak['time'] is not None:
                self.known_peaks[bb] = peak['time']
                truncation_chrom = chromatograms[0]
                full_chrom = self.full_compound_dict.get(tuple(building_blocks), [])[0][2]
                peak.update(self._calculate_measures(
                    full_chrom['deduplicated_counts'], truncation_chrom['deduplicated_counts'],
                    full_chrom['times'], truncation_chrom['times'],
                    measures
                ))
            results[bb] = peak
        return results

    def analyze_single_truncations(self, building_blocks: List[str], double_truncations: Dict[str, Dict], measures: Dict) -> Dict[Tuple[str, str], Dict]:
        results = {}
        for i in range(len(building_blocks)):
            for j in range(i+1, len(building_blocks)):
                bb_pair = (building_blocks[i], building_blocks[j])
                permutations_list = [
                    (bb_pair[0], bb_pair[1], self.n_1),
                    (bb_pair[0], self.n_1, bb_pair[1]),
                    (self.n_1, bb_pair[0], bb_pair[1])
                ]
                chromatograms = [chrom for perm in permutations_list for _, _, chrom in self.full_compound_dict.get(perm, [])]
                if not chromatograms:
                    print(f"No chromatograms found for single truncation {bb_pair}")
                    results[bb_pair] = {'peak': None, 'time': None, 'count': None, 'deduplicated_counts': None}
                    continue
                known_peaks = [double_truncations[bb]['time'] for bb in bb_pair if double_truncations.get(bb, {}).get('time') is not None]
                peak = self.peak_detector.find_consensus_peak(chromatograms, known_peaks, compound_type='single')
                if peak['time'] is not None:
                    truncation_chrom = chromatograms[0]
                    full_chrom = self.full_compound_dict.get(tuple(building_blocks), [])[0][2]
                    peak.update(self._calculate_measures(
                        full_chrom['deduplicated_counts'], truncation_chrom['deduplicated_counts'],
                        full_chrom['times'], truncation_chrom['times'],
                        measures
                    ))
                results[bb_pair] = peak
        return results

    def _calculate_measures(self, counts1, counts2, times1, times2, measures):
        # Implement measure calculations here
        return {}
