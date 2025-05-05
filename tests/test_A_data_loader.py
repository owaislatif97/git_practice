import unittest
import pandas as pd
import os
from unittest.mock import patch, mock_open
import sys
sys.path.append('.')  # Adds the project directory to the path
from src.forecast.A_data_loader import DataLoader

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.loader = DataLoader()

        # Create a mock DataFrame for testing
        self.mock_df = pd.DataFrame({
            'Time': ['2023-01-01 00:00', '2023-01-01 01:00'],
            'Power': [100.0, 120.0],
            'temperature_2m': [10.0, 11.0],
            'windspeed_10m': [5.0, 6.0]
        })
        self.mock_df['Time'] = pd.to_datetime(self.mock_df['Time'])

    @patch('pandas.read_csv')
    def test_load_data(self, mock_read_csv):
        """Test the load_data method"""
        # Configure the mock
        mock_read_csv.return_value = self.mock_df

        # Call the method to test
        result = self.loader.load_data(1)

        # Verify the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)  # Two rows as in mock
        mock_read_csv.assert_called_once_with('inputs/Location1.csv')

    def test_get_data_info_no_data(self):
        """Test get_data_info when no data is loaded"""
        # Reset the loader
        self.loader = DataLoader()

        # Call the method
        result = self.loader.get_data_info()

        # Verify the result
        self.assertEqual(result, "No data loaded")

    def test_get_data_info_with_data(self):
        """Test get_data_info with loaded data"""
        # Set up the loader with data
        self.loader.data = self.mock_df

        # Call the method
        result = self.loader.get_data_info()

        # Verify the result
        self.assertIsInstance(result, dict)
        self.assertEqual(result['rows'], 2)
        self.assertIn('Time', result['columns'])
        self.assertIn('Power', result['columns'])
        self.assertEqual(len(result['time_range']), 2)

if __name__ == '__main__':
    unittest.main()