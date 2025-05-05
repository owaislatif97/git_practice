import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.forecast.A_data_loader import DataLoader
from src.forecast.B_preprocessing import DataPreprocessor
from src.forecast.C_models import ModelTrainer
from src.forecast.D_predictor import Predictor
from src.forecast.E_evaluation import Evaluator

def display_column_info(df):
    """Display information about the columns in the dataframe"""
    print("\nDataset features:")
    for col in df.columns:
        if col not in ['Time', 'Power']:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            print(f"- {col}: range [{min_val:.2f} to {max_val:.2f}], mean: {mean_val:.2f}")

def get_user_input():
    """Collect user input for weather parameters with proper column names"""
    print("\n=== Enter parameters for one-hour ahead prediction ===")
    parameters = {}

    # Use the exact column names from the dataset
    parameters['temperature'] = float(input("Temperature at 2m (°C): "))
    parameters['humidity'] = float(input("Relative Humidity at 2m (%): "))
    parameters['dewpoint'] = float(input("Dewpoint at 2m (°C): "))
    parameters['windspeed_10m'] = float(input("Wind Speed at 10m (m/s): "))
    parameters['windspeed_100m'] = float(input("Wind Speed at 100m (m/s): "))
    parameters['winddirection_10m'] = float(input("Wind Direction at 10m (degrees): "))
    parameters['winddirection_100m'] = float(input("Wind Direction at 100m (degrees): "))
    parameters['windgusts'] = float(input("Wind Gusts at 10m (m/s): "))

    # If you want to include current power for persistence model
    current_power = input("\nCurrent Power (optional, for persistence model): ")
    if current_power.strip():
        parameters['current_power'] = float(current_power)

    return parameters

def get_location_choice():
    """Prompt user to select a location for data analysis"""
    while True:
        print("\n=== Choose a location for wind data analysis ===")
        print("1: Location 1")
        print("2: Location 2")
        print("3: Location 3")
        print("4: Location 4")

        choice = input("Enter your choice (1-4): ")

        try:
            location_index = int(choice)
            if 1 <= location_index <= 4:
                return location_index
            else:
                print("Error: Please enter a number between 1 and 4.")
        except ValueError:
            print("Error: Please enter a valid number.")

def interactive_prediction_mode(predictor, df):
    """Run interactive prediction mode where user can input parameters"""
    # First, display info about the features to help the user
    display_column_info(df)

    while True:
        parameters = get_user_input()

        # Make predictions with all models
        try:
            predictions = predictor.predict_all_from_parameters(parameters)

            # Display results
            print("\n=== One-Hour Ahead Power Predictions ===")
            for model_name, prediction in predictions.items():
                print(f"{model_name.upper()} Model: {prediction:.4f}")
        except Exception as e:
            print(f"Error making prediction: {str(e)}")

        # Ask if user wants to make another prediction
        if input("\nMake another prediction? (y/n): ").lower() != 'y':
            break

def main():
    # Create output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Get user choice for location
    location_index = get_location_choice()

    # Step 1: Load data from selected location
    print(f"\nStep 1: Loading data from Location {location_index}...")
    loader = DataLoader()
    df = loader.load_data(location_index)

    data_info = loader.get_data_info()
    print(f"Total dataset size: {data_info['rows']} rows")
    print(f"Dataset time range: {data_info['time_range'][0]} to {data_info['time_range'][1]}")

    # Step 2: Preprocess the data
    print("\nStep 2: Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, time_test = preprocessor.preprocess(df)
    print("✅ Preprocessing complete!")

    # Generate correlation heatmap
    print("\nGenerating correlation heatmap...")
    heatmap_path = preprocessor.correlation_heatmap(output_dir=output_dir)
    print(f"✅ Correlation heatmap saved to {heatmap_path}")

    # Define how many points to visualize
    last_n_points = 500
    print(f"\nWe'll visualize the last {last_n_points} data points")

    # Step 3: Implement TASK 1 - One hour ahead prediction with persistence model
    print("\n🔮 TASK 1: One-hour ahead prediction with Persistence Model...")
    predictor = Predictor()
    y_true_persist, y_pred_persist, time_persist = predictor.predict_persistence(df)

    # Evaluate persistence model
    evaluator = Evaluator(output_dir=output_dir)
    mae, rmse, r2 = evaluator.evaluate_model(y_true_persist, y_pred_persist, model_name="persistence")
    evaluator.print_metrics("persistence")

    # Step 4: Implement TASK 2 - One hour ahead prediction with ML models
    print("\n🔮 TASK 2: One-hour ahead prediction with ML Models...")

    # Prepare data for one-hour ahead prediction
    X_train_1h, X_test_1h, y_train_1h, y_test_1h, time_test_1h = preprocessor.prepare_features_for_one_hour_ahead()

    # Train models for one-hour ahead prediction
    trainer = ModelTrainer()
    models = trainer.train_all_models(X_train_1h, y_train_1h)

    # Make predictions with trained models
    predictor = Predictor(models)
    # Pass the scaler and feature names to the predictor
    predictor.set_scaler(preprocessor.scaler, preprocessor.X_raw.columns.tolist())
    predictions = predictor.predict_with_models(X_test_1h)

    # Add persistence predictions to all predictions
    predictions['persistence'] = y_pred_persist[-len(y_test_1h):]

    # Evaluate ML models
    for name, y_pred in predictions.items():
        evaluator.evaluate_model(y_test_1h, y_pred, model_name=name)

    evaluator.print_metrics()

    # Step 5: Implement TASK 3 - Plot predictions for a specific period
    print("\n🔮 TASK 3: Plotting one-hour ahead predictions for Last 500 points...")

    # Plot predictions for last n points
    evaluator.plot_predictions(time_test_1h, y_test_1h, predictions, last_n_points)

    print("\n✅ All tasks completed successfully!")

    # Step 6: Interactive prediction mode
    print("\n🔮 TASK 4: Interactive one-hour ahead predictions...")
    interactive_mode = input("\nWould you like to enter interactive prediction mode? (y/n): ")
    if interactive_mode.lower() == 'y':
        interactive_prediction_mode(predictor, df)

if __name__ == "__main__":
    main()