import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
import sys
sys.path.append('.')  # Adds the project directory to the path
from src.forecast.D_predictor import Predictor

class TestPredictor(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        # Create mock models
        self.mock_rf = MagicMock()
        self.mock_rf.predict.return_value = np.array([90, 100, 110])

        self.mock_nn = MagicMock()
        self.mock_nn.predict.return_value = np.array([85, 95, 105])

        self.models = {
            'random_forest': self.mock_rf,
            'neural_network': self.mock_nn
        }

        # Create predictor with mock models
        self.predictor = Predictor(self.models)

        # Create mock data
        self.mock_df = pd.DataFrame({
            'Time': pd.date_range(start='2023-01-01', periods=5, freq='h'),
            'Power': [100, 110, 120, 130, 140]
        })

        # Create mock test data
        self.X_test = np.random.rand(3, 5)

        # Set up scaler
        self.mock_scaler = MagicMock()
        self.mock_scaler.transform.return_value = np.random.rand(1, 5)
        self.feature_names = ['temperature_2m', 'windspeed_10m', 'windspeed_100m',
                             'relativehumidity_2m', 'dewpoint_2m']

    def test_predict_with_models(self):
        """Test making predictions with all models"""
        result = self.predictor.predict_with_models(self.X_test)

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result['random_forest'], np.array([90, 100, 110]))
        np.testing.assert_array_equal(result['neural_network'], np.array([85, 95, 105]))

        self.mock_rf.predict.assert_called_once_with(self.X_test)
        self.mock_nn.predict.assert_called_once_with(self.X_test)

    def test_predict_persistence(self):
        """Test persistence model predictions"""
        y_true, y_pred, time = self.predictor.predict_persistence(self.mock_df)

        # Verify the results
        self.assertEqual(len(y_true), 4)  # One less due to shift
        self.assertEqual(len(y_pred), 4)
        self.assertEqual(len(time), 4)

        # Fix: Create expected series with float dtype to match
        expected = pd.Series([100.0, 110.0, 120.0, 130.0], name='Power')

        pd.testing.assert_series_equal(y_pred, expected)

    def test_get_predictions(self):
        """Test retrieving predictions"""
        self.predictor.predictions = {
            'random_forest': np.array([90, 100, 110]),
            'neural_network': np.array([85, 95, 105])
        }

        all_preds = self.predictor.get_predictions()
        self.assertEqual(len(all_preds), 2)

        rf_preds = self.predictor.get_predictions('random_forest')
        np.testing.assert_array_equal(rf_preds, np.array([90, 100, 110]))

    def test_set_and_use_scaler(self):
        """Test setting and using the scaler"""
        self.predictor.set_scaler(self.mock_scaler, self.feature_names)

        self.assertEqual(self.predictor.scaler, self.mock_scaler)
        self.assertEqual(self.predictor.feature_names, self.feature_names)

    def test_predict_from_parameters(self):
        """Test predicting from user parameters"""
        self.predictor.set_scaler(self.mock_scaler, self.feature_names)

        params = {
            'temperature': 15.0,
            'humidity': 70.0,
            'dewpoint': 10.0,
            'windspeed_10m': 5.0,
            'windspeed_100m': 8.0,
            'winddirection_10m': 180.0,
            'winddirection_100m': 185.0,
            'windgusts': 7.0
        }

        result = self.predictor.predict_from_parameters(params, 'random_forest')

        self.assertEqual(result, 90)  # First value from mock_rf predictions
        self.mock_scaler.transform.assert_called_once()
        self.mock_rf.predict.assert_called()

    def test_predict_all_from_parameters(self):
        """Test predicting from parameters with all models"""
        self.predictor.set_scaler(self.mock_scaler, self.feature_names)
        self.predictor.predict_from_parameters = MagicMock()
        self.predictor.predict_from_parameters.side_effect = [90, 85]

        params = {
            'temperature': 15.0,
            'humidity': 70.0,
            'dewpoint': 10.0,
            'windspeed_10m': 5.0,
            'windspeed_100m': 8.0,
            'current_power': 100.0
        }

        result = self.predictor.predict_all_from_parameters(params)

        self.assertEqual(len(result), 3)
        self.assertEqual(result['random_forest'], 90)
        self.assertEqual(result['neural_network'], 85)
        self.assertEqual(result['persistence'], 100.0)

if __name__ == '__main__':
    unittest.main()
