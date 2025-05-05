"""Module for data preprocessing and feature engineering."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Handles data preprocessing including feature engineering and normalization."""

    def __init__(self):
        """Initialize preprocessor with empty attributes."""
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.time_test = None
        self.X_raw = None
        self.y_raw = None
        self.time_column = None
        self.df_processed = None

    def _add_time_features(self, df):
        """Add time-based features to the dataframe."""
        df['hour'] = df['Time'].dt.hour
        df['day_of_week'] = df['Time'].dt.dayofweek
        df['month'] = df['Time'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        return df

    def _add_lag_features(self, df, cols, lags=None):
        """Add lagged versions of selected columns."""
        if lags is None:
            lags = [1, 2]
        for col in cols:
            for lag in lags:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        return df

    def _add_interaction_features(self, df):
        """Add interaction features between existing variables."""
        df['wind_speed_ratio'] = df['windspeed_100m'] / (df['windspeed_10m'] + 0.1)
        df['temp_humidity_interaction'] = df['temperature_2m'] * df['relativehumidity_2m']
        return df

    def _add_wind_direction_features(self, df):
        """Convert wind direction from degrees to sine and cosine components."""
        df['winddir_10m_sin'] = np.sin(np.deg2rad(df['winddirection_10m']))
        df['winddir_10m_cos'] = np.cos(np.deg2rad(df['winddirection_10m']))
        df['winddir_100m_sin'] = np.sin(np.deg2rad(df['winddirection_100m']))
        df['winddir_100m_cos'] = np.cos(np.deg2rad(df['winddirection_100m']))
        return df

    def _add_moving_averages(self, df, cols, windows=None):
        """Add moving averages of selected columns with specified window sizes."""
        if windows is None:
            windows = [3]
        for col in cols:
            for window in windows:
                df[f'{col}_ma{window}'] = df[col].rolling(window=window).mean()
        return df

    def correlation_heatmap(self, df=None, output_dir="outputs"):
        """
        Plot heatmap of correlations between all numeric features.

        Args:
            df: DataFrame to use (if None, uses stored processed data)
            output_dir: Directory to save the heatmap

        Returns:
            Path to saved heatmap
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        if df is None:
            if self.df_processed is None:
                raise ValueError("No data available. Either provide a DataFrame or run preprocess() first.")
            df = self.df_processed

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['float64', 'int64'])

        # Create correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()

        # Save heatmap
        filepath = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(filepath)
        plt.show()
        plt.close()

        return filepath

    def preprocess(self, df, test_size=0.2, random_state=42):
        """
        Preprocess input data: add features, scale, and split into train/test sets.

        Args:
            df: Input DataFrame with raw data
            test_size: Proportion of data for testing set
            random_state: Random seed for reproducibility

        Returns:
            tuple: X_train, X_test, y_train, y_test, time_test
        """
        df = df.copy()

        # Store original time
        df['Time'] = pd.to_datetime(df['Time'])

        # Add engineered features
        df = self._add_time_features(df)
        df = self._add_interaction_features(df)
        df = self._add_wind_direction_features(df)
        df = self._add_moving_averages(df, ['temperature_2m', 'windspeed_10m'])

        # Add lag features
        weather_columns = ['temperature_2m', 'windspeed_10m']
        df = self._add_lag_features(df, weather_columns)

        # Clean data
        df = df.dropna().reset_index(drop=True)

        # Store processed dataframe for visualization
        self.df_processed = df.copy()

        # Split into features and target
        self.X_raw = df.drop(columns=['Time', 'Power'])
        self.y_raw = df['Power']
        self.time_column = df['Time']

        # Scale features
        X_scaled = self.scaler.fit_transform(self.X_raw)

        # Split into train and test sets
        split_result = train_test_split(
            X_scaled, self.y_raw, self.time_column,
            test_size=test_size, random_state=random_state
        )
        self.X_train, self.X_test, self.y_train, self.y_test, _, self.time_test = split_result

        return self.X_train, self.X_test, self.y_train, self.y_test, self.time_test

    def prepare_features_for_one_hour_ahead(self):
        """
        Prepare features and labels for one-hour ahead forecasting.

        Returns:
            tuple: X_train_1h_scaled, X_test_1h_scaled, y_train_1h, y_test_1h, time_test_1h
        """
        data = pd.concat([
            self.time_column.reset_index(drop=True),
            pd.DataFrame(self.X_raw, columns=self.X_raw.columns),
            pd.Series(self.y_raw, name='Power')
        ], axis=1)

        data = data.sort_values('Time').reset_index(drop=True)

        # Shift target for one-hour ahead forecasting
        y_target = data['Power'].shift(-1)
        X_features = data.drop(columns=['Time', 'Power'])

        # Drop last row (no target for it)
        X_features, y_target = X_features[:-1], y_target[:-1]
        time_target = data['Time'][:-1]

        # Split into train/test
        train_size = int(0.8 * len(X_features))
        X_train_1h = X_features.iloc[:train_size]
        y_train_1h = y_target.iloc[:train_size]
        X_test_1h = X_features.iloc[train_size:]
        y_test_1h = y_target.iloc[train_size:]
        time_test_1h = time_target.iloc[train_size:]

        # Scale features
        X_train_1h_scaled = self.scaler.transform(X_train_1h)
        X_test_1h_scaled = self.scaler.transform(X_test_1h)

        return X_train_1h_scaled, X_test_1h_scaled, y_train_1h, y_test_1h, time_test_1h