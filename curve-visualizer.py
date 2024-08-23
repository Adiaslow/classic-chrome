class CurveVisualizer:
    """Class for visualizing data."""

    @staticmethod
    def plot_deduplicated_counts(df: pd.DataFrame, building_blocks: List[str], measures: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Plot deduplicated counts and return measures dataframe."""
        if len(building_blocks) != 3:
            raise ValueError("Please provide exactly 3 building blocks.")

        matched_df = CompoundMatcher.optimize_compound_matching(df, building_blocks)

        unique_groups = matched_df['group'].unique()
        color_map = {group: rgb2hex(plt.colormaps['tab10'](i)) for i, group in enumerate(unique_groups)}

        fig, axs = plt.subplots(3, 2, figsize=(30, 45))
        axs = axs.flatten()

        CurveVisualizer._set_plot_style()

        full_compound = matched_df[matched_df['group'] == '_'.join(building_blocks)].iloc[0]
        full_times, full_counts = CurveAnalysis.normalize_curves(full_compound['Times'], full_compound['Deduplicated Counts'])
        common_times = np.linspace(min(full_times), max(full_times), 1000)
        full_counts_interp = CurveAnalysis.interpolate_data(full_times, full_counts, common_times)

        null_compound = matched_df[matched_df['group'] == 'Null'].iloc[0]
        null_times, null_counts = CurveAnalysis.normalize_curves(null_compound['Times'], null_compound['Deduplicated Counts'])
        null_counts_interp = CurveAnalysis.interpolate_data(null_times, null_counts, common_times)

        plot_order = [
            f'{building_blocks[0]}',
            f'{building_blocks[1]}',
            f'{building_blocks[2]}',
            f'{building_blocks[0]}_{building_blocks[1]}',
            f'{building_blocks[0]}_{building_blocks[2]}',
            f'{building_blocks[1]}_{building_blocks[2]}',
        ]

        all_results = {}
        for i, group_name in enumerate(plot_order):
            if i < len(axs):
                group_df = matched_df[matched_df['group'] == group_name]
                if not group_df.empty:
                    all_results[group_name] = CurveVisualizer._plot_group(axs[i], group_name, group_df, common_times,
                                                                     full_counts_interp, null_counts_interp, measures)
                else:
                    CurveVisualizer._plot_empty_group(axs[i], group_name)

        null_results = CurveVisualizer._calculate_null_results(common_times, full_counts_interp, null_counts_interp, measures)
        all_results['AgxNull'] = null_results

        fig.suptitle(f"Deduplicated Counts for {' '.join(building_blocks)}", fontsize=36, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, right=0.9, hspace=0.3, wspace=0.3)
        plt.show()

        return CurveVisualizer.display_measures_dataframe(all_results)

    @staticmethod
    def _plot_group(ax: plt.Axes, group_name: str, group_df: pd.DataFrame, common_times: np.ndarray,
                    full_counts_interp: np.ndarray, null_counts_interp: np.ndarray,
                    measures: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Plot a single group and calculate measures."""
        results = {cat: {name: {} for name in measures[cat]} for cat in measures}

        ax.plot(common_times, full_counts_interp, label="Full Compound", linewidth=2, color='black', zorder=-4)
        ax.plot(common_times, null_counts_interp, label='AgxNull_AgxNull_AgxNull', linewidth=2, color='gray', linestyle='--', zorder=98)

        for _, compound in group_df.iterrows():
            times, counts = CurveAnalysis.normalize_curves(compound['Times'], compound['Deduplicated Counts'])
            counts_interp = CurveAnalysis.interpolate_data(times, counts, common_times)

            label = '_'.join(compound['compound_key'])
            if label != "Full Compound":
                ax.plot(common_times, counts_interp, label=label, linewidth=2)

            for cat, cat_measures in measures.items():
                for name, func in cat_measures.items():
                    if name == 'Area Between Curves':
                        results[cat][name][label] = func(common_times, full_counts_interp, counts_interp)
                    else:
                        results[cat][name][label] = func(full_counts_interp, counts_interp)

        CurveVisualizer._set_axes_properties(ax, group_name)
        return results

    @staticmethod
    def _set_plot_style():
        """Set the plot style."""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial'] + plt.rcParams['font.sans-serif']

        SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 24, 32, 36

        plt.rc('font', size=SMALL_SIZE)
        plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE)
        plt.rc('xtick', labelsize=SMALL_SIZE)
        plt.rc('ytick', labelsize=SMALL_SIZE)
        plt.rc('legend', fontsize=SMALL_SIZE)
        plt.rc('figure', titlesize=BIGGER_SIZE)

    @staticmethod
    def _set_axes_properties(ax: plt.Axes, group_name: str):
        """Set properties for plot axes."""
        ax.set_xlabel('Time')
        ax.set_ylabel('Normalized Deduplicated Counts')
        ax.set_title(f"Normalized Deduplicated Counts for {group_name}")
        ax.legend()
        ax.tick_params(axis='both', which='major')
        ax.grid(False)

    @staticmethod
    def _plot_empty_group(ax: plt.Axes, group_name: str):
        """Plot an empty group."""
        ax.text(0.5, 0.5, f"No data for {group_name}", ha='center', va='center',
                fontsize=60, fontweight='bold', fontname='Arial')
        ax.set_title(f"Normalized Deduplicated Counts for {group_name}",
                     fontsize=60, fontweight='bold', fontname='Arial')

    @staticmethod
    def _calculate_null_results(common_times: np.ndarray, full_counts_interp: np.ndarray,
                                null_counts_interp: np.ndarray, measures: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Calculate measures for the null compound."""
        null_results = {cat: {name: {} for name in measures[cat]} for cat in measures}
        for cat, cat_measures in measures.items():
            for name, func in cat_measures.items():
                if name == 'Area Between Curves':
                    null_results[cat][name]['AgxNull_AgxNull_AgxNull'] = func(common_times, full_counts_interp, null_counts_interp)
                else:
                    null_results[cat][name]['AgxNull_AgxNull_AgxNull'] = func(full_counts_interp, null_counts_interp)
        return null_results

    @staticmethod
    def display_measures_dataframe(all_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]) -> pd.DataFrame:
        """Create a DataFrame from the calculated measures."""
        all_measures = []

        for group_name, results in all_results.items():
            for category, measures in results.items():
                for measure_name, compounds in measures.items():
                    for compound, value in compounds.items():
                        all_measures.append({
                            'Group': group_name,
                            'Category': category,
                            'Measure': measure_name,
                            'Compound': compound,
                            'Value': value
                        })

        df = pd.DataFrame(all_measures)

        pivot_df = df.pivot_table(
            values='Value',
            index=['Group', 'Compound'],
            columns=['Category', 'Measure'],
            aggfunc='first'
        )

        pivot_df = pivot_df.sort_index(axis=1)
        pivot_df = pivot_df.fillna('-')

        for col in pivot_df.columns:
            if pivot_df[col].dtype == 'float64':
                pivot_df[col] = pivot_df[col].apply(lambda x: f'{x:.4f}' if x != '-' else x)

        return pivot_df