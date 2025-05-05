import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def create_ensemble_weights(predictions_dict, equal_weights=True):
    """
    Create weights for ensemble models.

    Args:
        predictions_dict: Dictionary of {model_name: predictions}
        equal_weights: Whether to assign equal weights (True) or not

    Returns:
        Dictionary of {model_name: weight}
    """
    if equal_weights:
        weights = {name: 1.0/len(predictions_dict) for name in predictions_dict.keys()}
    else:
        # Initialize with equal weights, will be updated later
        weights = {name: 1.0/len(predictions_dict) for name in predictions_dict.keys()}

    return weights

def ensemble_predict(predictions_dict, weights):
    """
    Create weighted ensemble prediction.

    Args:
        predictions_dict: Dictionary of {model_name: predictions}
        weights: Dictionary of {model_name: weight}

    Returns:
        Weighted average prediction
    """
    if not predictions_dict:
        raise ValueError("No predictions available.")

    # Create weighted ensemble prediction
    ensemble_pred = np.zeros_like(list(predictions_dict.values())[0])

    for name, pred in predictions_dict.items():
        ensemble_pred += weights[name] * pred

    return ensemble_pred

def update_weights_by_performance(predictions_dict, y_true, lookback_window=48):
    """
    Update model weights based on recent performance.
    Weights are inversely proportional to MAE over recent window.

    Args:
        predictions_dict: Dictionary of {model_name: predictions}
        y_true: True values for the most recent predictions
        lookback_window: Number of most recent timesteps to evaluate

    Returns:
        Updated weights dictionary
    """
    if not predictions_dict:
        raise ValueError("No predictions available.")

    # Take only the most recent points
    recent_true = y_true[-lookback_window:]
    recent_errors = {}

    # Calculate MAE for each model over the recent window
    for name, pred in predictions_dict.items():
        recent_pred = pred[-lookback_window:]
        recent_errors[name] = mean_absolute_error(recent_true, recent_pred)

    # Convert errors to weights (smaller error = larger weight)
    # Add small constant to avoid division by zero
    total_inverse_error = sum(1.0 / (err + 1e-10) for err in recent_errors.values())
    weights = {name: (1.0 / (err + 1e-10)) / total_inverse_error
              for name, err in recent_errors.items()}

    return weights



def plot_ensemble_comparison(time_test, y_true, predictions_dict, weights, ensemble_pred,
                           last_n_points=500, title="Ensemble Model Comparison"):
    """
    Plot comparison of all models including ensemble.

    Args:
        time_test: Time points for x-axis
        y_true: True values
        predictions_dict: Dictionary of {model_name: predictions}
        weights: Dictionary of {model_name: weight}
        ensemble_pred: Ensemble prediction
        last_n_points: Number of points to plot
        title: Plot title
    """
    # Take only the last n points
    times = time_test[-last_n_points:]
    trues = y_true[-last_n_points:]
    ensembles = ensemble_pred[-last_n_points:]

    # Create plot
    plt.figure(figsize=(14, 7))
    plt.plot(times, trues, 'k-', label='Actual Power', linewidth=2)

    # Plot individual model predictions
    colors = ['b-', 'g-', 'c-', 'm-', 'y-']
    color_idx = 0

    for name, pred in predictions_dict.items():
        model_preds = pred[-last_n_points:]
        color = colors[color_idx % len(colors)]
        plt.plot(times, model_preds, color, label=f'{name.capitalize()} (weight: {weights[name]:.2f})',
                alpha=0.4)
        color_idx += 1

    # Plot ensemble prediction
    plt.plot(times, ensembles, 'r-', label='Ensemble Prediction', linewidth=2)

    plt.title(title, fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Power Output', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    filename = 'ensemble_model_comparison.png'
    plt.savefig(filename)
    plt.show()
    print(f"📈 Plot saved as {filename}")