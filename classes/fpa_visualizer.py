import matplotlib.pyplot as plt
from typing import List, Dict, Any
from .fpa_data_preprocessor import FPADataPreprocessor

class FPAVisualizer:
    def __init__(self, data_preprocessor: 'FPADataPreprocessor'):
        self.data_preprocessor = data_preprocessor
        self.n_1 = data_preprocessor.n_1
        self.full_compound_dict = data_preprocessor.full_compound_dict
        self.annotation_color = 'black'  # Set a consistent color for all annotations


    def plot_compounds(self, results: Dict, building_blocks: List[str], measures: Dict = None) -> plt.Figure:
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(3, 3)

        full_null_compound = self._get_full_null_compound()

        self._plot_main_compound(fig.add_subplot(gs[0, :]), results, building_blocks, full_null_compound)

        double_truncations = results.get('double_truncations', {})
        single_truncations = results.get('single_truncations', {})

        def safe_get_time(x):
            if x[1] is None:
                return float('-inf')
            if isinstance(x[1], dict):
                return x[1].get('time', float('-inf'))
            print(f"Unexpected structure in truncation data: {x}")
            return float('-inf')

        # Filter out None values and sort
        valid_double_truncations = [(k, v) for k, v in double_truncations.items() if v is not None]
        valid_single_truncations = [(k, v) for k, v in single_truncations.items() if v is not None]

        sorted_double_truncations = sorted(valid_double_truncations, key=safe_get_time)
        sorted_single_truncations = sorted(valid_single_truncations, key=safe_get_time)

        dt_label_mapping = {bb: f"DT{i+1}" for i, (bb, _) in enumerate(sorted_double_truncations)}

        for i, (bb, data) in enumerate(sorted_double_truncations):
            if i < 3:  # Ensure we don't exceed the grid
                self._plot_truncation(fig.add_subplot(gs[1, i]), data, bb, i+1, 'DT', full_null_compound)

        for i, ((bb1, bb2), data) in enumerate(sorted_single_truncations):
            if i < 3:  # Ensure we don't exceed the grid
                relevant_double_truncations = {dt_label_mapping[bb]: double_truncations[bb] for bb in [bb1, bb2] if bb in double_truncations and double_truncations[bb] is not None}
                self._plot_truncation(fig.add_subplot(gs[2, i]), data, f"{bb1}_{bb2}", i+1, 'ST', full_null_compound, relevant_double_truncations)

        plt.tight_layout()
        return fig

    @lru_cache(maxsize=None)
    def _get_full_null_compound(self) -> Dict:
        full_null_key = (self.n_1, self.n_1, self.n_1)
        return next((chrom for _, _, chrom in self.full_compound_dict.get(full_null_key, [])), None)

    def _plot_main_compound(self, ax: plt.Axes, results: Dict, building_blocks: List[str], full_null_compound: Dict) -> None:
        self.building_blocks = building_blocks
        full_key = tuple(building_blocks)
        full_compound = next((chrom for _, _, chrom in self.full_compound_dict.get(full_key, [])), None)
        full_compound_data = results.get('full', {})
        full_compound_success = full_compound_data.get('success', False)
        full_compound_time = full_compound_data.get('time')

        if full_compound:
            ax.plot(full_compound['times'], full_compound['deduplicated_counts'], label='-'.join(building_blocks), linewidth=3)
        if full_null_compound:
            ax.plot(full_null_compound['times'], full_null_compound['deduplicated_counts'],
                    color='black', linestyle='--', linewidth=3, alpha=1, label='-'.join([self.n_1]*3), zorder=-10)

        truncation_data = self._prepare_truncation_data(results)

        # Plot truncations
        self._plot_retention_times(ax, None, truncation_data, max_y=ax.get_ylim()[1])

        # Plot the full compound only if the reaction was successful
        if full_compound_success and full_compound_time is not None:
            self._plot_retention_times(ax, full_compound_time, {}, main_label="FC", max_y=ax.get_ylim()[1], is_main=True)
        else:
            ax.text(0.5, 0.95, "Full Compound Reaction Failed",
                    ha='center', va='top', transform=ax.transAxes,
                    color='red', fontweight='bold', fontsize=12)

        if full_compound:
            x_min = full_compound['times'][0]
            x_max = full_compound['times'][-1]
            y_min = 0
            y_max = np.round((ax.get_ylim()[1] + 100) / 100) * 100

            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])

        ax.set_title(f"Full Compound (FC): {'-'.join(building_blocks)}")
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Deduplicated Counts')
        ax.spines[['right', 'top']].set_visible(False)
        ax.legend(loc='upper right', fontsize=10)

    def _plot_truncation(self, ax: plt.Axes, data: Dict, bb: str, number: int, truncation_type: str, full_null_compound: Dict, relevant_double_truncations: Dict = None) -> None:
        chromatograms = self._get_chromatograms(bb, truncation_type)
        title = f"{truncation_type}{number}: {bb}"
        self._plot_single_compound(ax, data, chromatograms, title, relevant_double_truncations, full_null_compound, f"{truncation_type}{number}")

    def _plot_single_compound(self, ax: plt.Axes, peak_data: Dict, chromatograms: List[Tuple[Tuple[str, ...], Dict]],
                              title: str, relevant_double_truncations: Dict = None,
                              full_null_compound: Dict = None, annotation_label: str = None) -> None:

        x_min = 0
        x_max = 0
        y_min = 0
        max_y = 0
        for compound, chromatogram in chromatograms:
            times, deduplicated_counts = np.array(chromatogram['times']), np.array(chromatogram['deduplicated_counts'])
            if len(times) > 0 and len(deduplicated_counts) > 0:
                x_min = times[0]
                x_max = times[-1]
                y_min = min(0, deduplicated_counts[0])
                max_y = max(max_y, np.max(deduplicated_counts))
                ax.plot(times, deduplicated_counts, alpha=1, label='-'.join(bb if bb != self.n_1 else self.n_1 for bb in compound), linewidth=2.5)

        if full_null_compound:
            ax.plot(full_null_compound['times'], full_null_compound['deduplicated_counts'],
                    color='black', linestyle='--', linewidth=2.5, alpha=1, label='-'.join([self.n_1]*3), zorder=-10)

        ax.set_title(title.replace('_', '-'))
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Deduplicated Counts')

        y_max = np.round((max_y + 100) / 100) * 100

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

        # Combine the current peak data with relevant double truncations
        all_retention_times = {annotation_label: peak_data}

        # Plot relevant double truncations first
        if relevant_double_truncations:
            self._plot_retention_times(ax, None, relevant_double_truncations, max_y=y_max)

        # Then plot the main peak
        self._plot_retention_times(ax, None, {annotation_label: peak_data}, max_y=y_max, is_main=True)

        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)

    def _plot_retention_times(self, ax: plt.Axes, main_rt: Optional[float], other_rts: Dict, main_label: str = "FC", max_y: float = None, is_main: bool = False) -> None:
        if isinstance(other_rts, dict):
            for label, data in other_rts.items():
                if isinstance(data, dict):
                    rt = data.get('time', data.get('peak_data', {}).get('time'))
                    if rt is not None:
                        self._add_retention_time_annotation(ax, rt, label, is_main=is_main, max_y=max_y, curve_label=label)

        if main_rt is not None:
            self._add_retention_time_annotation(ax, main_rt, main_label, is_main=True, max_y=max_y, curve_label='-'.join(self.building_blocks))

    def _add_retention_time_annotation(self, ax: plt.Axes, time: float, label: str, is_main: bool = False, max_y: float = None, curve_label: str = None) -> None:
        if max_y is None:
            max_y = ax.get_ylim()[1]

        # Find the y-value at the retention time for the specific curve
        y_value = self._find_y_value_for_curve(ax, time, curve_label)

        # Draw vertical line from the curve to the top of the plot
        ax.vlines(x=time, ymin=y_value, ymax=y_value + 10, color='gray' if not is_main else 'gray', linestyle='-', linewidth=2, alpha=0.2, zorder=-1)

        # Add text annotation above the peak
        ax.text(time, y_value + 10, f"{label}\n{time:.2f}",
                rotation=0, verticalalignment='bottom', horizontalalignment='center',
                color='black' if not is_main else 'black', fontweight='bold', fontsize=10,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white", alpha=0.6)])

    def _find_y_value_for_curve(self, ax: plt.Axes, time: float, curve_label: str) -> float:
        for line in ax.get_lines():
            if line.get_label() == curve_label:
                x_data = line.get_xdata()
                y_data = line.get_ydata()
                # Find the closest x-value to the given time
                idx = np.argmin(np.abs(x_data - time))
                return y_data[idx]

        # If the specific curve is not found, fall back to the previous behavior
        return max(line.get_ydata()[np.argmin(np.abs(line.get_xdata() - time))] for line in ax.get_lines())


    def _prepare_truncation_data(self, results: Dict) -> Dict[str, Dict]:
        double_truncations = results.get('double_truncations', {})
        single_truncations = results.get('single_truncations', {})
        truncation_data = {f"DT{i+1}": {'peak_data': data, 'original_title': bb} for i, (bb, data) in enumerate(double_truncations.items())}
        truncation_data.update({f"ST{i+1}": {'peak_data': data, 'original_title': f"{bb1}_{bb2}"} for i, ((bb1, bb2), data) in enumerate(single_truncations.items())})
        return dict(sorted(truncation_data.items(), key=lambda x: x[1]['peak_data'].get('time') if x[1]['peak_data'].get('time') is not None else float('-inf')))

    @lru_cache(maxsize=128)
    def _get_chromatograms(self, bb: str, truncation_type: str) -> List[Tuple[Tuple[str, ...], Dict]]:
        if truncation_type == 'DT':
            permutations = [(bb, self.n_1, self.n_1), (self.n_1, bb, self.n_1), (self.n_1, self.n_1, bb)]
        else:  # ST
            bb1, bb2 = bb.split('_')
            permutations = [(bb1, bb2, self.n_1), (bb1, self.n_1, bb2), (self.n_1, bb1, bb2)]
        return [(perm, chrom) for perm in permutations for _, _, chrom in self.full_compound_dict.get(perm, [])]

    def show_plot(self):
        plt.show()
