class CurveManipulator:
    """Class for manipulating and morphing curves."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.curve_analysis = CurveAnalysis()
        self.compound_matcher = CompoundMatcher()

    def morph_curves(self, compound: List[str], k: int = 5):
        """Morph curves for a given compound."""
        if len(compound) != 3:
            raise ValueError("Compound must be a list of exactly 3 building blocks.")

        matched_df = self.compound_matcher.optimize_compound_matching(self.df, compound)

        full_compound = matched_df[matched_df['group'] == '_'.join(compound)].iloc[0]
        null_compound = matched_df[matched_df['group'] == 'Null'].iloc[0]

        truncations = [
            f"{compound[0]}",
            f"{compound[1]}",
            f"{compound[2]}",
            f"{compound[0]}_{compound[1]}",
            f"{compound[0]}_{compound[2]}",
            f"{compound[1]}_{compound[2]}"
        ]

        fig, axs = plt.subplots(len(truncations), k, figsize=(8*k, 6*len(truncations)))
        fig.suptitle(f"Morphing Truncations to Full Compound: {' '.join(compound)}", fontsize=16, fontweight='bold', fontname='Arial')

        for i, truncation in enumerate(truncations):
            truncation_compound = matched_df[matched_df['group'] == truncation].iloc[0]
            self._plot_morphing_row(axs[i], full_compound, truncation_compound, null_compound, k, truncation)

        plt.tight_layout()
        plt.subplots_adjust(top=0.95, right=0.9, hspace=0.5, wspace=0.4)
        plt.show()


    def _plot_morphing_row(self, axs, full_compound: pd.Series, truncation_compound: pd.Series,
                           null_compound: pd.Series, k: int, truncation: str):
        """Plot a row of morphing curves."""
        full_times, full_counts = self.curve_analysis.normalize_curves(full_compound['Times'], full_compound['Deduplicated Counts'])
        trunc_times, trunc_counts = self.curve_analysis.normalize_curves(truncation_compound['Times'], truncation_compound['Deduplicated Counts'])
        null_times, null_counts = self.curve_analysis.normalize_curves(null_compound['Times'], null_compound['Deduplicated Counts'])

        common_times = np.linspace(min(full_times.min(), trunc_times.min(), null_times.min()),
                                   max(full_times.max(), trunc_times.max(), null_times.max()), 1000)

        full_counts_interp = self.curve_analysis.interpolate_data(full_times, full_counts, common_times)
        trunc_counts_interp = self.curve_analysis.interpolate_data(trunc_times, trunc_counts, common_times)
        null_counts_interp = self.curve_analysis.interpolate_data(null_times, null_counts, common_times)

        full_label = f"{full_compound['BB1 Name']}_{full_compound['BB2 Name']}_{full_compound['BB3 Name']}"
        trunc_label = '_'.join(bb for bb in [truncation_compound['BB1 Name'], truncation_compound['BB2 Name'], truncation_compound['BB3 Name']] if bb != 'AgxNull')
        null_label = "AgxNull_AgxNull_AgxNull"

        for j in range(k):
            alpha = j / (k - 1)
            morphed_counts = self._morph_time_series(trunc_counts_interp, full_counts_interp, alpha)

            ax = axs[j] if len(axs.shape) == 1 else axs[j]
            ax.plot(common_times, full_counts_interp, label=full_label, color='black')
            ax.plot(common_times, morphed_counts, label=f'{trunc_label} (α={alpha:.2f})', color='red')
            ax.plot(common_times, null_counts_interp, label=null_label, color='black', linestyle='--')

            self._set_plot_properties(ax, truncation, alpha)

            distances = self._calculate_distances(full_counts_interp, morphed_counts, null_counts_interp)
            self._add_distance_annotation(ax, distances)

        self._set_font_properties(ax)

    def _morph_time_series(self, ts1: np.ndarray, ts2: np.ndarray, alpha: float) -> np.ndarray:
        """Morph between two time series."""
        return (1 - alpha) * ts1 + alpha * ts2

    def _calculate_distances(self, full: np.ndarray, morphed: np.ndarray, null: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate distances between curves."""
        distances = {
            "Full-Truncation": {
                "Earth Mover's Distance": self.curve_analysis.earth_movers_distance(full, morphed),
                "Euclidean Distance": self.curve_analysis.euclidean_distance(full, morphed),
                "Jensen-Shannon Distance": self.curve_analysis.jensen_shannon_distance(full, morphed),
                "Chi-Squared Distance": self.curve_analysis.chi_square_distance(full, morphed)
            },
            "Full-Null": {
                "Earth Mover's Distance": self.curve_analysis.earth_movers_distance(full, null),
                "Euclidean Distance": self.curve_analysis.euclidean_distance(full, null),
                "Jensen-Shannon Distance": self.curve_analysis.jensen_shannon_distance(full, null),
                "Chi-Squared Distance": self.curve_analysis.chi_square_distance(full, null)
            },
            "Truncation-Null": {
                "Earth Mover's Distance": self.curve_analysis.earth_movers_distance(morphed, null),
                "Euclidean Distance": self.curve_analysis.euclidean_distance(morphed, null),
                "Jensen-Shannon Distance": self.curve_analysis.jensen_shannon_distance(morphed, null),
                "Chi-Squared Distance": self.curve_analysis.chi_square_distance(morphed, null)
            },
        }
        return distances

    def _add_distance_annotation(self, ax: plt.Axes, distances: Dict[str, Dict[str, float]]):
        """Add distance annotations to the right of the plot."""
        annotation = "Distances:\n"
        for pair, metrics in distances.items():
            annotation += f"{pair}:\n"
            for metric, value in metrics.items():
                annotation += f"  {metric}:\n    {value:.4f}\n"
            annotation += "\n"
        annotation = annotation[:-2]
        ax.annotate(annotation, xy=(1.05, 0.5), xycoords='axes fraction', fontsize='xx-small',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    va='center', ha='left')

    def _set_plot_properties(self, ax: plt.Axes, truncation: str, alpha: float):
        """Set properties for individual plots."""
        ax.set_title(f"{truncation} (α={alpha:.2f})", fontsize=14, fontproperties=arial_bold)
        ax.set_xlabel('Time', fontsize=12, fontproperties=arial_bold)
        ax.set_ylabel('Normalized Counts', fontsize=12, fontproperties=arial_bold)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(False)
        ax.legend(fontsize=10, prop=arial_bold)

    def _set_font_properties(self, ax: plt.Axes):
        """Set font properties for all text elements in the figure."""
        for text in ax.get_figure().findobj(match=lambda x: isinstance(x, plt.Text)):
            text.set_fontname('Arial')
            text.set_fontweight('bold')
