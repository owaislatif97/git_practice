import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
sys.path.append('.')  # Adds the project directory to the path
from src.forecast.E_evaluation import Evaluator

class TestEvaluator(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = Evaluator()

        # Create mock data
        np.random.seed(42)  # For reproducibility
        self.y_true = np.array([100, 110, 120, 130, 140])
        self.y_pred_rf = np.array([105, 112, 118, 133, 142])  # Random Forest predictions
        self.y_pred_nn = np.array([103, 115, 125, 128, 138])  # Neural Network predictions
        self.time_test = pd.Series(pd.date_range(start='2023-01-01', periods=5, freq='h'))

    def test_evaluate_model(self):
        """Test model evaluation metrics calculation"""
        mae, rmse, r2 = self.evaluator.evaluate_model(self.y_true, self.y_pred_rf, model_name='random_forest')

        # Verify metrics were calculated
        self.assertIsInstance(mae, float)
        self.assertIsInstance(rmse, float)
        self.assertIsInstance(r2, float)

        # Verify metrics are stored
        self.assertIn('random_forest', self.evaluator.metrics)
        self.assertEqual(self.evaluator.metrics['random_forest']['mae'], mae)

        # Verify metrics are reasonable
        self.assertLess(mae, 5)  # MAE should be small for our mock data
        self.assertGreater(r2, 0.9)  # R² should be high for our mock data

    def test_print_metrics(self):
        """Test printing of metrics (no actual assertions, just for coverage)"""
        # Add metrics for multiple models
        self.evaluator.evaluate_model(self.y_true, self.y_pred_rf, model_name='random_forest')
        self.evaluator.evaluate_model(self.y_true, self.y_pred_nn, model_name='neural_network')

        # Just call the method to make sure it doesn't crash
        with patch('builtins.print') as mock_print:
            self.evaluator.print_metrics()
            # Verify print was called multiple times
            self.assertGreater(mock_print.call_count, 5)

        # Test printing metrics for a specific model
        with patch('builtins.print') as mock_print:
            self.evaluator.print_metrics('random_forest')
            # Verify print was called
            self.assertGreater(mock_print.call_count, 3)

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_predictions(self, mock_show, mock_savefig, mock_plot, mock_figure):
        """Test plotting predictions (with mocked matplotlib)"""
        # Setup predictions dictionary
        predictions = {
            'random_forest': self.y_pred_rf,
            'neural_network': self.y_pred_nn,
            'persistence': np.array([95, 105, 115, 125, 135])
        }

        # Call the method
        self.evaluator.plot_predictions(self.time_test, self.y_true, predictions, last_n_points=None)

        # Verify functions were called
        mock_figure.assert_called()
        self.assertGreater(mock_plot.call_count, 3)  # At least one call for each model
        mock_savefig.assert_called()
        mock_show.assert_called()

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_predictions_with_last_n(self, mock_show, mock_savefig, mock_plot, mock_figure):
        """Test plotting with last_n_points filtering"""
        # Setup predictions dictionary
        predictions = {
            'random_forest': self.y_pred_rf,
            'neural_network': self.y_pred_nn
        }

        # Call the method with last_n_points=3
        self.evaluator.plot_predictions(self.time_test, self.y_true, predictions, last_n_points=3)

        # Verify functions were called
        mock_figure.assert_called()
        self.assertGreater(mock_plot.call_count, 2)  # At least one call for each model
        mock_savefig.assert_called()
        mock_show.assert_called()

if __name__ == '__main__':
    unittest.main()