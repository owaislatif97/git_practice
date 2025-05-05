"""Module for training machine learning models."""
from typing import Any, Dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV


class ModelTrainer:
    """Handles training and optimization of machine learning models."""

    def __init__(self):
        """Initialize with empty models dictionary."""
        self.models: Dict[str, Any] = {}

    def _tune_model(self, model, param_grid, X_train, y_train, n_iter=10,
                   cv=3, random_state=42, model_name='model'):
        """
        Tune model hyperparameters using RandomizedSearchCV.

        Args:
            model: Base model to tune
            param_grid: Parameter grid to search
            X_train: Training features
            y_train: Training targets
            n_iter: Number of parameter settings sampled
            cv: Cross-validation folds
            random_state: Random seed
            model_name: Name for the model

        Returns:
            The best estimator found
        """
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            verbose=1,
            random_state=random_state,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        print(f"Best parameters for {model_name}: {search.best_params_}")
        self.models[model_name] = search.best_estimator_
        return search.best_estimator_

    def train_random_forest(self, X_train, y_train, random_state=42):
        """
        Train an optimized Random Forest regression model.

        Args:
            X_train: Training features
            y_train: Training targets
            random_state: Random seed

        Returns:
            Trained Random Forest model
        """
        basic_rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=random_state
        )

        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10],
            'max_features': ['sqrt']
        }
        return self._tune_model(basic_rf, param_grid, X_train, y_train, n_iter=5, model_name='rf')

    def train_neural_network(self, X_train, y_train, random_state=42):
        """
        Train an optimized Neural Network regression model.

        Args:
            X_train: Training features
            y_train: Training targets
            random_state: Random seed

        Returns:
            Trained Neural Network model
        """
        basic_nn = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=200,
            early_stopping=True,
            random_state=random_state
        )

        param_grid = {
            'hidden_layer_sizes': [(64, 32), (32, 16)],
            'alpha': [0.001, 0.01],
            'max_iter': [200, 300]
        }
        return self._tune_model(basic_nn, param_grid, X_train, y_train, n_iter=4, model_name='nn')

    def train_all_models(self, X_train, y_train):
        """
        Train all ML models and return dictionary of trained models.

        Args:
            X_train: Training features
            y_train: Training targets

        Returns:
            dict: Dictionary of trained models
        """
        print("Training Random Forest...")
        rf_model = self.train_random_forest(X_train, y_train)

        print("Training Neural Network...")
        nn_model = self.train_neural_network(X_train, y_train)

        return {
            'random_forest': rf_model,
            'neural_network': nn_model
        }