"""Module for making predictions with trained models."""
import numpy as np
import pandas as pd


class Predictor:
    """Handles making predictions with trained models."""

    def __init__(self, models=None):
        """
        Initialize predictor with optional pre-trained models.

        Args:
            models: Dictionary of trained models
        """
        self.models = models or {}
        self.predictions = {}
        self.scaler = None
        self.feature_names = None

    def set_scaler(self, scaler, feature_names):
        """
        Set the scaler for preprocessing new inputs.

        Args:
            scaler: Fitted StandardScaler
            feature_names: List of feature names
        """
        self.scaler = scaler
        self.feature_names = feature_names

    def predict_with_models(self, X_test):
        """
        Make predictions using all trained models.

        Args:
            X_test: Test features

        Returns:
            dict: Dictionary of predictions for each model
        """
        self.predictions = {}
        for name, model in self.models.items():
            self.predictions[name] = model.predict(X_test)
        return self.predictions

    def predict_persistence(self, df):
        """
        Implement the persistence model (next hour = current hour).

        Args:
            df: DataFrame with Power column

        Returns:
            tuple: y_true, y_pred, time
        """
        y_true = df['Power'].iloc[1:].reset_index(drop=True)
        y_pred = df['Power'].shift(1).iloc[1:].reset_index(drop=True)
        time = df['Time'].iloc[1:].reset_index(drop=True)

        self.predictions['persistence'] = y_pred
        return y_true, y_pred, time

    def get_predictions(self, model_name=None):
        """
        Return predictions for a specific model or all models.

        Args:
            model_name: Name of specific model (optional)

        Returns:
            Predictions array or dictionary
        """
        if model_name:
            return self.predictions.get(model_name)
        return self.predictions

    def predict_from_parameters(self, parameters, model_name='rf'):
        """
        Make one-hour ahead prediction from user-provided parameters.

        Args:
            parameters: Dictionary with weather parameters
            model_name: Which model to use for prediction

        Returns:
            float: Predicted power value
        """
        if self.scaler is None or self.feature_names is None:
            raise ValueError("Scaler or feature names not set. Call set_scaler() first.")

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Available models: {list(self.models.keys())}")

        # Create dataframe with expected feature names
        input_df = pd.DataFrame(0, index=[0], columns=self.feature_names)

        # Map parameters to feature names
        feature_mapping = {
            'windspeed_10m': 'windspeed_10m',
            'windspeed_100m': 'windspeed_100m',
            'temperature': 'temperature_2m',
            'humidity': 'relativehumidity_2m',
            'dewpoint': 'dewpoint_2m',
            'windgusts': 'windgusts_10m'
        }

        # Fill with provided values
        for user_param, feature_name in feature_mapping.items():
            if user_param in parameters and feature_name in input_df.columns:
                input_df[feature_name] = parameters[user_param]

        # Add calculated features
        input_df['wind_speed_ratio'] = input_df['windspeed_100m'] / (input_df['windspeed_10m'] + 0.1)
        input_df['temp_humidity_interaction'] = input_df['temperature_2m'] * input_df['relativehumidity_2m']

        # Handle wind direction conversion
        if 'winddirection_10m' in parameters:
            wind_dir_10m = parameters['winddirection_10m']
            rad_10m = np.deg2rad(wind_dir_10m)
            if 'winddir_10m_sin' in input_df.columns:
                input_df['winddir_10m_sin'] = np.sin(rad_10m)
            if 'winddir_10m_cos' in input_df.columns:
                input_df['winddir_10m_cos'] = np.cos(rad_10m)

        if 'winddirection_100m' in parameters:
            wind_dir_100m = parameters['winddirection_100m']
            rad_100m = np.deg2rad(wind_dir_100m)
            if 'winddir_100m_sin' in input_df.columns:
                input_df['winddir_100m_sin'] = np.sin(rad_100m)
            if 'winddir_100m_cos' in input_df.columns:
                input_df['winddir_100m_cos'] = np.cos(rad_100m)

        # Scale features
        input_scaled = self.scaler.transform(input_df)

        # Make prediction
        prediction = self.models[model_name].predict(input_scaled)[0]

        return prediction

    def predict_all_from_parameters(self, parameters):
        """
        Make predictions with all available models.

        Args:
            parameters: Dictionary with weather parameters

        Returns:
            dict: Dictionary of predictions for each model
        """
        result = {}
        for model_name in self.models.keys():
            result[model_name] = self.predict_from_parameters(parameters, model_name)

        # Add persistence prediction if current power is provided
        if 'current_power' in parameters:
            result['persistence'] = parameters['current_power']

        return result