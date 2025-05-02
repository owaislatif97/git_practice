import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
import matplotlib.pyplot as plt

def create_bayesian_model(alpha=1e-10, n_restarts_optimizer=10, kernel=None):
    """
    Create a Bayesian model using Gaussian Process Regression.
    
    Args:
        alpha: Value added to the diagonal of the kernel matrix (noise)
        n_restarts_optimizer: Number of restarts for kernel parameter optimization
        kernel: Custom kernel (if None, uses a default composite kernel)
    
    Returns:
        Gaussian Process Regressor model
    """
    if kernel is None:
        # Create a composite kernel
        # Matern kernel is good for physical processes
        # WhiteKernel accounts for noise in measurements
        kernel = ConstantKernel() * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1)
    
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        n_restarts_optimizer=n_restarts_optimizer,
        normalize_y=True,
        random_state=42
    )
    
    return model

def train_bayesian_model(model, X_train, y_train):
    """
    Train the Bayesian model.
    
    Args:
        model: GaussianProcessRegressor model
        X_train: Training features
        y_train: Training targets
    
    Returns:
        Trained model
    """
    model.fit(X_train, y_train)
    return model

def predict_with_uncertainty(model, X_test):
    """
    Make predictions with uncertainty estimates.
    
    Args:
        model: Trained GaussianProcessRegressor
        X_test: Test features
    
    Returns:
        y_pred: Mean predictions
        y_std: Standard deviations
    """
    return model.predict(X_test, return_std=True)

def predict_with_intervals(model, X_test, confidence=0.95):
    """
    Predict with confidence intervals.
    
    Args:
        model: Trained GaussianProcessRegressor
        X_test: Test features
        confidence: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        y_pred: Mean predictions
        y_lower: Lower confidence bound
        y_upper: Upper confidence bound
    """
    y_pred, y_std = predict_with_uncertainty(model, X_test)
    
    # Calculate Z-score for the given confidence level
    z_score = np.abs(np.percentile(np.random.normal(0, 1, 10000), 
                                   100 * (1 - (1 - confidence) / 2)))
    
    # Calculate confidence intervals
    y_lower = y_pred - z_score * y_std
    y_upper = y_pred + z_score * y_std
    
    return y_pred, y_lower, y_upper

def plot_bayesian_prediction(model, time_test, y_test, X_test, 
                           confidence=0.95, title="Bayesian Wind Power Prediction"):
    """
    Plot predictions with uncertainty bands.
    
    Args:
        model: Trained GaussianProcessRegressor
        time_test: Time points for x-axis
        y_test: True values for comparison
        X_test: Test features for prediction
        confidence: Confidence level for intervals
        title: Plot title
    """
    y_pred, y_lower, y_upper = predict_with_intervals(model, X_test, confidence)
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_test, y_test, 'k-', label='Actual Power', linewidth=2)
    plt.plot(time_test, y_pred, 'r-', label='Bayesian Prediction', alpha=0.7)
    plt.fill_between(time_test, y_lower, y_upper, color='r', alpha=0.2, 
                    label=f'{int(confidence*100)}% Confidence Interval')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Power Output', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    filename = 'bayesian_prediction_with_uncertainty.png'
    plt.savefig(filename)
    plt.show()
    print(f"📈 Plot saved as {filename}")