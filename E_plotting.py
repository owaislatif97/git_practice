import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(time_test, y_test, predictions, start_time=None, end_time=None, last_n_hours=None):
    """
    Plots and saves predictions vs actuals.
    Can filter using either a time window or last N hours/points.
    
    Parameters:
    - time_test: Series of timestamps
    - y_test: True values (array or Series)
    - predictions: Dict of model_name -> predicted values
    - start_time, end_time: Optional time range to filter
    - last_n_hours: Optional, if provided, only the last N points will be plotted
    """
    for name, y_pred in predictions.items():
        # Convert to arrays for easier handling
        times = np.array(time_test)
        trues = np.array(y_test)
        preds = np.array(y_pred)
        
        # Take last N points if specified
        if last_n_hours is not None:
            n = min(last_n_hours, len(times))  # Make sure we don't exceed array length
            times = times[-n:]
            trues = trues[-n:]
            preds = preds[-n:]
            
            # Update start/end times for title
            start_time = times[0]
            end_time = times[-1]
        elif start_time is not None and end_time is not None:
            # Filter by time range if start/end times provided
            mask = (times >= start_time) & (times <= end_time)
            times = times[mask]
            trues = trues[mask]
            preds = preds[mask]
        else:
            # Default to using all data
            start_time = times[0]
            end_time = times[-1]
            
        # Create plot
        plt.figure(figsize=(12, 5))
        plt.plot(times, trues, label='Actual Power', alpha=0.8)
        plt.plot(times, preds, label=f'{name.upper()} Prediction', alpha=0.8)
        plt.xlabel("Time")
        plt.ylabel("Power")
        plt.title(f"{name.upper()} Prediction vs Actual\n{start_time} to {end_time}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        filename = f"{name}_predicted_vs_actual.png"
        plt.savefig(filename)
        plt.show()
        print(f"📈 Plot saved as {filename}")
        print(f"    Time range: {start_time} to {end_time}")
        print(f"    Number of points: {len(times)}")