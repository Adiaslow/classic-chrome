import numpy as np
from typing import List, Tuple, Dict, Set
from scipy.stats import wasserstein_distance
from fpa_data_preprocessor import FPADataPreprocessor
from fpa_peak_detector import FPAPeakDetector
from fpa_truncation_analyzer import FPATruncationAnalyzer
from fpa_visualizer import FPAVisualizer
from curve_analysis import CurveAnalysis

class FPACompoundAnalyzer:
    def __init__(self, data_preprocessor: FPADataPreprocessor, peak_detector: FPAPeakDetector, truncation_analyzer: FPATruncationAnalyzer):
        self.visualizer = FPAVisualizer(data_preprocessor)
        self.data_preprocessor = data_preprocessor
        self.peak_detector = peak_detector
        self.truncation_analyzer = truncation_analyzer
        self.curve_analyzer = CurveAnalysis()
        self.n_1 = data_preprocessor.n_1
        self.full_compound_dict = data_preprocessor.full_compound_dict
        self.full_null_peak = self._analyze_full_null_compound()

    def analyze_compound(self, building_blocks: List[str], generate_plots: bool = False, measures: Dict = None) -> Dict:
        try:
            original_bbs = [bb if bb != 'null' else self.n_1 for bb in building_blocks]
            non_null_bbs = [bb for bb in original_bbs if bb != self.n_1]

            results = {}

            if len(non_null_bbs) == 0:
                results['full'] = self._analyze_null_compound(original_bbs)
            elif len(non_null_bbs) == 1:
                results['double_truncations'] = self.truncation_analyzer.analyze_double_truncations(non_null_bbs, measures)
            elif len(non_null_bbs) == 2:
                results['double_truncations'] = self.truncation_analyzer.analyze_double_truncations(non_null_bbs, measures)
                results['single_truncations'] = self.truncation_analyzer.analyze_single_truncations(non_null_bbs, results['double_truncations'], measures)
            else:
                results['double_truncations'] = self.truncation_analyzer.analyze_double_truncations(non_null_bbs, measures)
                results['single_truncations'] = self.truncation_analyzer.analyze_single_truncations(non_null_bbs, results['double_truncations'], measures)
                results['full'] = self._analyze_full_compound(non_null_bbs, results['double_truncations'], results['single_truncations'], measures)

            print("Analysis results:")
            for key, value in results.items():
                print(f"{key}: {value}")

            if 'full' not in results:
                print(f"Warning: 'full' key missing from results for compound {original_bbs}")
                results['full'] = {'error': f"No full compound data for {original_bbs}"}

            if generate_plots:
                try:
                    self._plot_compounds(results, original_bbs, measures)
                except Exception as plot_error:
                    print(f"Error during plotting: {str(plot_error)}")
                    import traceback
                    print(traceback.format_exc())

            return self._build_hypothesis(results, original_bbs)
        except Exception as e:
            print(f"Error in analyze_compound: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {'error': str(e), 'traceback': traceback.format_exc()}

    def _analyze_full_compound(self, building_blocks: List[str], double_truncations: Dict[str, Dict], single_truncations: Dict[Tuple[str, str], Dict], measures: Dict) -> Dict:
        key = tuple(building_blocks)
        chromatograms = [chrom for _, _, chrom in self.full_compound_dict.get(key, [])]
        if not chromatograms:
            print(f"No chromatograms found for full compound {building_blocks}")
            return {'peak': None, 'time': None, 'count': None, 'deduplicated_counts': None}

        known_peaks = [
            *[double_truncations[bb]['time'] for bb in building_blocks if double_truncations.get(bb, {}).get('time') is not None],
            *[single_truncations[pair]['time'] for pair in permutations(building_blocks, 2) if single_truncations.get(pair, {}).get('time') is not None]
        ]

        peak_data = self.peak_detector.find_consensus_peak(chromatograms, known_peaks, compound_type='full')

        if peak_data['time'] is not None:
            full_chrom = self.full_compound_dict.get(key, [])[0][2]
            null_chrom = self.full_compound_dict.get((self.n_1, self.n_1, self.n_1), [])[0][2]
            peak_data.update(self._calculate_measures(
                full_chrom['deduplicated_counts'], null_chrom['deduplicated_counts'],
                full_chrom['times'], null_chrom['times'],
                measures
            ))

        return peak_data

    def _build_hypothesis(self, results: Dict, building_blocks: List[str]) -> Dict:
        if 'full' not in results:
            return {'error': f"'full' key missing from results for compound {building_blocks}"}

        hypothesis = {
            'full_compound_peak': None,
            'successful_reactions': [],
            'failed_reactions': []
        }

        non_null_bbs = [bb for bb in building_blocks if bb != self.n_1]

        self._check_double_truncations(results, non_null_bbs, hypothesis)
        self._check_single_truncations(results, hypothesis)
        self._check_full_compound(results, non_null_bbs, hypothesis)

        hypothesis['successful_reactions'] = ', '.join(hypothesis['successful_reactions'])
        hypothesis['failed_reactions'] = ', '.join(hypothesis['failed_reactions'])

        return hypothesis

    def _check_double_truncations(self, results: Dict, non_null_bbs: List[str], hypothesis: Dict) -> None:
        if 'double_truncations' not in results:
            return

        for bb in non_null_bbs:
            if bb in results['double_truncations'] and results['double_truncations'][bb]['time'] is not None:
                if results['double_truncations'][bb]['time'] > self.full_null_peak:
                    hypothesis['successful_reactions'].append(bb)

    def _check_single_truncations(self, results: Dict, hypothesis: Dict) -> None:
        if 'single_truncations' not in results:
            return

        for truncation, data in results['single_truncations'].items():
            if data['time'] is not None and data['time'] > self.full_null_peak:
                hypothesis['successful_reactions'].append('-'.join(truncation))

    def _check_full_compound(self, results: Dict, non_null_bbs: List[str], hypothesis: Dict) -> None:
        full_peak = results['full']
        if full_peak['time'] is None:
            hypothesis['failed_reactions'].append('-'.join(non_null_bbs))
            return

        max_truncation_peak_time = self._get_max_truncation_peak_time(results)

        if max_truncation_peak_time is not None:
            if full_peak['time'] > max_truncation_peak_time and full_peak['time'] > self.full_null_peak:
                hypothesis['full_compound_peak'] = full_peak['time']
                hypothesis['successful_reactions'].append('-'.join(non_null_bbs))
            else:
                hypothesis['failed_reactions'].append('-'.join(non_null_bbs))
        else:
            if full_peak['time'] > self.full_null_peak:
                hypothesis['full_compound_peak'] = full_peak['time']
                hypothesis['successful_reactions'].append('-'.join(non_null_bbs))
            else:
                hypothesis['failed_reactions'].append('-'.join(non_null_bbs))

    def _get_max_truncation_peak_time(self, results: Dict) -> Optional[float]:
        if 'single_truncations' not in results:
            return None

        truncation_times = [data['time'] for data in results['single_truncations'].values() if data['time'] is not None]
        return max(truncation_times) if truncation_times else None

    def _compare_full_peak_with_truncation(self, full_peak: Dict, max_truncation_peak_time: float, non_null_bbs: List[str], hypothesis: Dict) -> None:
        if full_peak['time'] > max_truncation_peak_time and full_peak['time'] > self.full_null_peak:
            hypothesis['full_compound_peak'] = full_peak['time']
            hypothesis['successful_reactions'].append('-'.join(non_null_bbs))
        else:
            hypothesis['failed_reactions'].append('-'.join(non_null_bbs))

    def _check_full_peak_against_null(self, full_peak: Dict, non_null_bbs: List[str], hypothesis: Dict) -> None:
        if full_peak['time'] > self.full_null_peak:
            hypothesis['full_compound_peak'] = full_peak['time']
            hypothesis['successful_reactions'].append('-'.join(non_null_bbs))
        else:
            hypothesis['failed_reactions'].append('-'.join(non_null_bbs))

    def _determine_truncation_match(self, full_result: Dict, double_truncations: Dict[str, Dict], single_truncations: Dict[Tuple[str, str], Dict], building_blocks: List[str]) -> str:
        full_peak = full_result.get('time')
        if full_peak is None:
            return "No peak found for full compound"

        if full_peak <= self.full_null_peak:
            return "Peak not greater than full null compound"

        for i in range(len(building_blocks)):
            for j in range(i+1, len(building_blocks)):
                bb_pair = tuple(sorted([building_blocks[i], building_blocks[j]]))
                if bb_pair in single_truncations and single_truncations[bb_pair].get('time') == full_peak:
                    return f"Single: {'-'.join(bb_pair)}"

        for bb in building_blocks:
            if bb in double_truncations and double_truncations[bb].get('time') == full_peak:
                return f"Double: {bb}"

        return None

    def _analyze_single_compound(self, compound_data: Tuple[pd.Series, Dict, Dict]) -> Dict:
        row, double_truncations, single_truncations = compound_data
        building_blocks = [row['BB1 Name'], row['BB2 Name'], row['BB3 Name']]

        result = row.to_dict()
        result['Error'] = None

        try:
            full_analysis = self._analyze_full_compound(building_blocks, double_truncations, single_truncations, {})

            compound_double_truncations = {bb: double_truncations[bb] for bb in building_blocks if bb in double_truncations}

            compound_single_truncations = {}
            for i in range(len(building_blocks)):
                for j in range(i+1, len(building_blocks)):
                    bb_pair = tuple(sorted([building_blocks[i], building_blocks[j]]))
                    if bb_pair in single_truncations:
                        compound_single_truncations[bb_pair] = single_truncations[bb_pair]

            hypothesis = self._build_hypothesis({
                'full': full_analysis,
                'double_truncations': compound_double_truncations,
                'single_truncations': compound_single_truncations
            }, building_blocks)

            result['Cyclized RT (min)'] = full_analysis['time'] if full_analysis['time'] is not None else 0
            result['Successful Reactions'] = hypothesis.get('successful_reactions', [])
            result['Failed Reactions'] = hypothesis.get('failed_reactions', [])
            result['Truncation Match'] = self._determine_truncation_match(full_analysis, compound_double_truncations, compound_single_truncations, building_blocks)

        except Exception as e:
            result['Cyclized RT (min)'] = 0
            result['Successful Reactions'] = []
            result['Failed Reactions'] = [tuple(building_blocks)]
            result['Truncation Match'] = None
            result['Error'] = str(e)

        return result

    def _calculate_measures(self, counts1: np.ndarray, counts2: np.ndarray, times1: np.ndarray, times2: np.ndarray, measures: Dict) -> Dict[str, float]:
        results = {}

        try:
            counts1, counts2 = np.array(counts1), np.array(counts2)
            times1, times2 = np.array(times1), np.array(times2)

            start_time = max(times1[0], times2[0])
            end_time = min(times1[-1], times2[-1])
            common_times = np.linspace(start_time, end_time, max(len(counts1), len(counts2)))

            counts1_interp = self.curve_analyzer.interpolate_data(times1, counts1, common_times)
            counts2_interp = self.curve_analyzer.interpolate_data(times2, counts2, common_times)

            _, counts1_norm = self.curve_analyzer.normalize_curves(common_times, counts1_interp)
            _, counts2_norm = self.curve_analyzer.normalize_curves(common_times, counts2_interp)

            for category, category_measures in measures.items():
                for measure_name, measure_func in category_measures.items():
                    if measure_func is not None:
                        try:
                            if measure_name == 'Area Between Curves':
                                results[measure_name] = float(measure_func(common_times, counts1_norm, counts2_norm))
                            else:
                                results[measure_name] = float(measure_func(counts1_norm, counts2_norm))
                        except Exception as e:
                            print(f"Error calculating {measure_name}: {str(e)}")
                            results[measure_name] = None
            return results
        except Exception as e:
            # print(f"Error calculating measures: {str(e)}")
            return {}

    def _analyze_full_null_compound(self) -> float:
        null_compound = (self.n_1, self.n_1, self.n_1)
        chromatograms = [chrom for _, _, chrom in self.full_compound_dict.get(null_compound, [])]
        null_peak = self.peak_detector.find_consensus_peak(chromatograms, compound_type='null')
        return null_peak['time'] if null_peak['time'] is not None else 0

    def _plot_compounds(self, results: Dict, building_blocks: List[str], measures: Dict = None) -> None:
        fig = self.visualizer.plot_compounds(results, building_blocks, measures)
        self.visualizer.show_plot()

    def _analyze_null_compound(self, original_bbs: List[str]) -> Dict:
        # This method should be implemented to handle the case when all building blocks are null
        pass

    def _analyze_single_bb_compound(self, original_bbs: List[str], double_truncations: Dict[str, Dict], measures: Dict) -> Dict:
        # This method should be implemented to handle the case when there's only one non-null building block
        pass

    def _analyze_double_bb_compound(self, original_bbs: List[str], single_truncations: Dict[Tuple[str, str], Dict], measures: Dict) -> Dict:
        # This method should be implemented to handle the case when there are two non-null building blocks
        pass
