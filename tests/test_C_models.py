import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
sys.path.append('.')  # Adds the project directory to the path
from src.forecast.C_models import ModelTrainer

class TestModelTrainer(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.trainer = ModelTrainer()

        # Create mock training data
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.rand(100) * 100

    @patch('src.forecast.C_models.RandomizedSearchCV')
    def test_tune_model(self, mock_randomized_search):
        """Test the model tuning function"""
        # Create mock model and param grid
        mock_model = MagicMock()
        param_grid = {'param1': [1, 2, 3], 'param2': [4, 5, 6]}

        # Configure the mock
        mock_search_instance = MagicMock()
        mock_search_instance.best_params_ = {'param1': 2, 'param2': 5}
        mock_search_instance.best_estimator_ = MagicMock()
        mock_randomized_search.return_value = mock_search_instance

        # Call the method to test
        result = self.trainer._tune_model(mock_model, param_grid, self.X_train, self.y_train, model_name='test_model')

        # Verify the result
        self.assertEqual(result, mock_search_instance.best_estimator_)
        self.assertEqual(self.trainer.models['test_model'], mock_search_instance.best_estimator_)
        mock_randomized_search.assert_called_once()
        mock_search_instance.fit.assert_called_once_with(self.X_train, self.y_train)

    @patch('src.forecast.C_models.ModelTrainer._tune_model')
    @patch('src.forecast.C_models.RandomForestRegressor')
    def test_train_random_forest(self, mock_rf_class, mock_tune_model):
        """Test training a Random Forest model"""
        # Configure the mocks
        mock_rf_instance = MagicMock()
        mock_rf_class.return_value = mock_rf_instance

        mock_tuned_model = MagicMock()
        mock_tune_model.return_value = mock_tuned_model

        # Call the method to test
        result = self.trainer.train_random_forest(self.X_train, self.y_train)

        # Verify the result
        self.assertEqual(result, mock_tuned_model)
        mock_rf_class.assert_called_once()
        mock_tune_model.assert_called_once()

    @patch('src.forecast.C_models.ModelTrainer.train_random_forest')
    @patch('src.forecast.C_models.ModelTrainer.train_neural_network')
    def test_train_all_models(self, mock_train_nn, mock_train_rf):
        """Test training all models function"""
        # Configure the mocks
        mock_rf_model = MagicMock()
        mock_nn_model = MagicMock()
        mock_train_rf.return_value = mock_rf_model
        mock_train_nn.return_value = mock_nn_model

        # Call the method to test
        result = self.trainer.train_all_models(self.X_train, self.y_train)

        # Verify the result
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertEqual(result['random_forest'], mock_rf_model)
        self.assertEqual(result['neural_network'], mock_nn_model)
        mock_train_rf.assert_called_once_with(self.X_train, self.y_train)
        mock_train_nn.assert_called_once_with(self.X_train, self.y_train)

if __name__ == '__main__':
    unittest.main()