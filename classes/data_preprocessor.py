import pandas as pd
from typing import List, Tuple

class DataPreprocessor:
    """Class for preprocessing data."""

    @staticmethod
    def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataset.

        Args:
            df (pd.DataFrame): Raw dataset.

        Returns:
            pd.DataFrame: Preprocessed dataset.
        """
        filtered_df = df[df["lid"] != "DEL-0044"][["enumerated_smiles", "BB1 Name", "BB2 Name", "BB3 Name", "all_datapoints"]]

        trimmed_df = pd.DataFrame(columns=["SMILES", "BB1 Name", "BB2 Name", "BB3 Name", "Cyclized RT (min)"])
        trimmed_df["SMILES"] = filtered_df["enumerated_smiles"]
        trimmed_df["BB1 Name"] = filtered_df["BB1 Name"].fillna("AgxNull")
        trimmed_df["BB2 Name"] = filtered_df["BB2 Name"].fillna("AgxNull")
        trimmed_df["BB3 Name"] = filtered_df["BB3 Name"].fillna("AgxNull")
        trimmed_df["Cyclized RT (min)"] = 0


        processed_data = filtered_df["all_datapoints"].apply(DataPreprocessor._process_datapoints)
        trimmed_df["Times"] = processed_data.apply(lambda x: x[0])
        trimmed_df["Counts"] = processed_data.apply(lambda x: x[1])
        trimmed_df["Deduplicated Counts"] = processed_data.apply(lambda x: x[2])
        trimmed_df["Success"] = False
        return trimmed_df

    @staticmethod
    def _process_datapoints(datapoints: str) -> Tuple[List[float], List[float], List[float]]:
        """Process all datapoints for a row.

        Args:
            datapoints (str): String containing datapoints.

        Returns:
            Tuple[List[float], List[float], List[float]]: Processed times, counts, and deduplicated counts.
        """
        times, counts, deduplicated_counts = [], [], []
        for datapoint in datapoints.split(','):
            temp = datapoint.replace(":", " ").replace(";", " ").split()
            times.append(float(temp[0]) / 60 if temp else 0)
            counts.append(float(temp[1]) if len(temp) > 1 else 0)
            deduplicated_counts.append(float(temp[2]) if len(temp) > 2 else 0)

        # Sort the data based on times
        sorted_data = sorted(zip(times, counts, deduplicated_counts), key=lambda x: x[0])

        # Unzip the sorted data
        times, counts, deduplicated_counts = zip(*sorted_data)

        return list(times), list(counts), list(deduplicated_counts)
