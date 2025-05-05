"""Module for loading and initial data processing."""
import pandas as pd


class DataLoader:
    """Handles loading and initial processing of weather data."""

    def __init__(self):
        """Initialize with empty data attribute."""
        self.data = None
    def load_data(self, location_index):
        """
        Load CSV data for a specified site index (1-4).
        Args:
            location_index: Integer identifying the location (1-4)
        Returns:
            DataFrame: The loaded and initially processed data
        """
        path = f"inputs/Location{location_index}.csv"
        df = pd.read_csv(path)
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.dropna()
        self.data = df
        return self.data
    def get_data_info(self):
        """
        Return information about the loaded dataset.
        Returns:
            dict: Dataset information including rows, columns, time range,
            and missing values
        """
        if self.data is None:
            return "No data loaded"
        return {
            "rows": len(self.data),
            "columns": list(self.data.columns),
            "time_range": [self.data['Time'].min(), self.data['Time'].max()],
            "missing_values": self.data.isna().sum().sum()
        }