# forecast/evaluator.py
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import os

class Evaluator:
    """
    Handles evaluation of model predictions and visualization
    """

    def __init__(self, output_dir="outputs"):
        self.metrics = {}
        self.output_dir = output_dir
        # Create the outputs directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate_model(self, y_true, y_pred, model_name="Model"):
        """
        Evaluates model predictions and calculates MAE, RMSE, R²
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        self.metrics[model_name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

        return mae, rmse, r2

    def print_metrics(self, model_name=None):
        """
        Print evaluation metrics for a model or all models
        """
        if model_name:
            if model_name in self.metrics:
                m = self.metrics[model_name]
                print(f"\n{model_name.upper()} Model Results:")
                print(f"MAE : {m['mae']:.4f}")
                print(f"RMSE: {m['rmse']:.4f}")
                print(f"R²  : {m['r2']:.4f}")
            else:
                print(f"No metrics available for {model_name}")
        else:
            for name, m in self.metrics.items():
                print(f"\n{name.upper()} Model Results:")
                print(f"MAE : {m['mae']:.4f}")
                print(f"RMSE: {m['rmse']:.4f}")
                print(f"R²  : {m['r2']:.4f}")

    def plot_predictions(self, time_test, y_test, predictions, last_n_points=None):
        """
        Plots predictions vs actuals for all models with visually distinctive appearance
        """
        # Filter to last n points if specified
        if last_n_points:
            time_test = time_test.iloc[-last_n_points:]
            y_test = y_test[-last_n_points:]
            for model, preds in predictions.items():
                predictions[model] = preds[-last_n_points:]

        # Create a list of models to ensure consistent ordering
        model_names = list(predictions.keys())

        # COMPLETELY DIFFERENT APPROACH: Plot each model separately with VERY distinct styles
        plt.figure(figsize=(14, 7))

        # Plot actual power with black solid line
        plt.plot(time_test, y_test, 'k-', label='Actual Power', linewidth=2)

        # Plot models with maximally different appearances
        if 'rf' in model_names or 'random_forest' in model_names:
            rf_name = 'rf' if 'rf' in model_names else 'random_forest'
            # BLUE with TRIANGULAR markers
            plt.plot(time_test, predictions[rf_name], 'b-', marker='^', markevery=30,
                     label='Random Forest (1h ahead)', linewidth=1.5, markersize=6)

        if 'nn' in model_names or 'neural_network' in model_names:
            nn_name = 'nn' if 'nn' in model_names else 'neural_network'
            # GREEN with SQUARE markers
            plt.plot(time_test, predictions[nn_name], 'g-', marker='s', markevery=30,
                     label='Neural Network (1h ahead)', linewidth=1.5, markersize=6)

        if 'persistence' in model_names:
            # RED with DASHED line
            plt.plot(time_test, predictions['persistence'], 'r--',
                     label='Persistence Model (1h ahead)', linewidth=1.5)

        # Handle any other models with distinctive styles
        other_models = [m for m in model_names if m not in ['rf', 'nn', 'persistence', 'random_forest', 'neural_network']]
        colors = ['purple', 'orange', 'cyan', 'brown', 'lime']
        markers = ['o', 'x', 'd', 'p', '*']

        for i, model in enumerate(other_models):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            plt.plot(time_test, predictions[model], color=color, marker=marker, markevery=30,
                     label=f'{model.upper()} (1h ahead)', linewidth=1.5, markersize=6)

        plt.title('One-Hour Ahead Power Prediction', fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Power Output', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()

        # Save the combined plot to outputs folder
        combined_plot_path = os.path.join(self.output_dir, 'one_hour_ahead_prediction_comparison.png')
        plt.savefig(combined_plot_path)
        plt.show()
        print(f"✅ Plot saved as '{combined_plot_path}'")

        # Create individual plots for each model
        for model_name in model_names:
            plt.figure(figsize=(12, 6))
            plt.plot(time_test, y_test, 'k-', label='Actual Power', linewidth=2)

            # Apply the same distinctive styling to individual plots
            if model_name.lower() in ['rf', 'random_forest']:
                plt.plot(time_test, predictions[model_name], 'b-', marker='^', markevery=30,
                         label='Random Forest (1h ahead)', linewidth=1.5, markersize=6)
            elif model_name.lower() in ['nn', 'neural_network']:
                plt.plot(time_test, predictions[model_name], 'g-', marker='s', markevery=30,
                         label='Neural Network (1h ahead)', linewidth=1.5, markersize=6)
            elif model_name.lower() == 'persistence':
                plt.plot(time_test, predictions[model_name], 'r--',
                         label='Persistence Model (1h ahead)', linewidth=1.5)
            else:
                # For other models, use consistent styling
                idx = other_models.index(model_name)
                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]
                plt.plot(time_test, predictions[model_name], color=color, marker=marker, markevery=30,
                         label=f'{model_name.upper()} (1h ahead)', linewidth=1.5, markersize=6)

            plt.title(f'One-Hour Ahead Power Prediction - {model_name.upper()}', fontsize=16)
            plt.xlabel('Time', fontsize=14)
            plt.ylabel('Power Output', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            plt.tight_layout()

            # Save each individual plot to outputs folder
            filename = f'one_hour_ahead_{model_name.lower()}.png'
            file_path = os.path.join(self.output_dir, filename)
            plt.savefig(file_path)
            plt.show()
            print(f"✅ Plot saved as '{file_path}'")