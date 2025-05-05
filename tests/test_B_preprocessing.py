import unittest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
sys.path.append('.')  # Adds the project directory to the path
from src.forecast.B_preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = DataPreprocessor()

        # Create a mock DataFrame for testing
        dates = pd.date_range(start='2023-01-01', periods=10, freq='h')
        self.mock_df = pd.DataFrame({
            'Time': dates,
            'Power': np.random.rand(10) * 100,
            'temperature_2m': np.random.rand(10) * 30,
            'relativehumidity_2m': np.random.rand(10) * 100,
            'dewpoint_2m': np.random.rand(10) * 20,
            'windspeed_10m': np.random.rand(10) * 15,
            'windspeed_100m': np.random.rand(10) * 20,
            'winddirection_10m': np.random.rand(10) * 360,
            'winddirection_100m': np.random.rand(10) * 360,
            'windgusts_10m': np.random.rand(10) * 25
        })

        # Create a temporary directory for test outputs
        self.test_output_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests"""
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.test_output_dir)

    def test_add_time_features(self):
        """Test adding time-based features"""
        result = self.preprocessor._add_time_features(self.mock_df.copy())

        # Verify new columns were added
        self.assertIn('hour', result.columns)
        self.assertIn('day_of_week', result.columns)
        self.assertIn('month', result.columns)
        self.assertIn('is_weekend', result.columns)

        # Verify values make sense
        self.assertEqual(result['hour'][0], 0)  # First date is midnight
        self.assertTrue(all(result['is_weekend'].isin([0, 1])))

    def test_add_lag_features(self):
        """Test adding lag features"""
        cols = ['temperature_2m', 'windspeed_10m']
        result = self.preprocessor._add_lag_features(self.mock_df.copy(), cols, lags=[1, 2])

        # Verify new columns were added
        self.assertIn('temperature_2m_lag1', result.columns)
        self.assertIn('temperature_2m_lag2', result.columns)
        self.assertIn('windspeed_10m_lag1', result.columns)
        self.assertIn('windspeed_10m_lag2', result.columns)

        # Verify lag values are correct (after the lag positions)
        self.assertEqual(result['temperature_2m_lag1'][1], self.mock_df['temperature_2m'][0])
        self.assertEqual(result['windspeed_10m_lag2'][2], self.mock_df['windspeed_10m'][0])

    def test_add_wind_direction_features(self):
        """Test converting wind direction to sine and cosine components"""
        result = self.preprocessor._add_wind_direction_features(self.mock_df.copy())

        # Verify new columns were added
        self.assertIn('winddir_10m_sin', result.columns)
        self.assertIn('winddir_10m_cos', result.columns)
        self.assertIn('winddir_100m_sin', result.columns)
        self.assertIn('winddir_100m_cos', result.columns)

        # Verify values are in correct range (-1 to 1)
        self.assertTrue(all(result['winddir_10m_sin'].between(-1, 1)))
        self.assertTrue(all(result['winddir_10m_cos'].between(-1, 1)))

    def test_correlation_heatmap(self):
        """Test correlation heatmap generation"""
        # First run preprocess to populate the processed dataframe
        self.preprocessor.preprocess(self.mock_df)

        # Patch plt.show to prevent displaying plot during tests
        import matplotlib.pyplot as plt
        original_show = plt.show
        plt.show = lambda: None

        try:
            # Test that the heatmap is created and saved
            filepath = self.preprocessor.correlation_heatmap(output_dir=self.test_output_dir)
            self.assertTrue(os.path.exists(filepath))
            self.assertEqual(filepath, os.path.join(self.test_output_dir, "correlation_heatmap.png"))
        finally:
            # Restore original plt.show
            plt.show = original_show

    def test_preprocess_function(self):
        """Test the main preprocessing function"""
        X_train, X_test, y_train, y_test, time_test = self.preprocessor.preprocess(self.mock_df)

        # Verify outputs are created with correct types
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(X_test, np.ndarray)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(y_test, pd.Series)
        self.assertIsInstance(time_test, pd.Series)

        # Verify training and test set sizes
        self.assertEqual(len(X_train) + len(X_test), len(self.mock_df) - 2)  # -2 due to lag features

        # Verify scaler is fitted
        self.assertTrue(hasattr(self.preprocessor.scaler, 'mean_'))

if __name__ == '__main__':
    unittest.main()