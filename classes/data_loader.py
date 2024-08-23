import pandas as pd

class DataLoader:
    """Class for loading data."""

    @staticmethod
    def load_dataset(file_path: str) -> pd.DataFrame:
        """Load the dataset from a CSV file."""
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Only CSV and Excel files are supported.")
