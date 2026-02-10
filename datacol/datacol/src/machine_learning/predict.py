#!/usr/bin/env python
"""
Interactive stock price prediction script.
"""
import os
import sys
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from data_loader import DataLoader
from preprocessing import DataPreprocessor
from models import StockPredictionModels
import config


def interactive_predict_stock():
    """
    Interactive function to predict stock prices with user input
    """
    print("\n=== Interactive Stock Price Prediction ===\n")

    # Get user input for ticker
    ticker = input(f"Enter stock ticker symbol (default: {config.DEFAULT_TICKER}): ").strip()
    if not ticker:
        ticker = config.DEFAULT_TICKER

    # Get user input for days back
    while True:
        days_back_input = input(f"Enter number of days of historical data to use (default: 30): ").strip()
        if not days_back_input:
            days_back = 30
            break
        try:
            days_back = int(days_back_input)
            if days_back <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    # Get user input for prediction horizon
    while True:
        days_ahead_input = input(
            f"Enter number of days ahead to predict (default: {config.PREDICTION_HORIZON}): ").strip()
        if not days_ahead_input:
            prediction_days = config.PREDICTION_HORIZON
            break
        try:
            prediction_days = int(days_ahead_input)
            if prediction_days <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    # Get user input for model selection
    available_models = ["linear", "ridge", "lasso", "elasticnet", "randomforest", "xgboost"]
    print("\nAvailable models:", ", ".join(available_models))
    model_name = input("Enter model to use (leave blank for best model): ").strip().lower()
    if model_name and model_name not in available_models:
        print(f"Warning: Model '{model_name}' not recognized. Will try to use best model.")
        model_name = None

    model_dir = config.DEFAULT_MODELS_DIR

    # Make predictions
    predictions = predict_stock(
        ticker=ticker,
        days_back=days_back,
        model_name=model_name,
        prediction_days=prediction_days,
        model_dir=model_dir
    )

    if predictions:
        # Ask about saving predictions
        save_pred = input("\nSave predictions to a file? (y/n): ").strip().lower() == 'y'
        if save_pred:
            output_file = input("Enter output file path (leave blank for default): ").strip()
            save_predictions(predictions, output_file if output_file else None)

        # Ask about plotting
        plot_pred = input("\nGenerate a plot of the predictions? (y/n): ").strip().lower() == 'y'
        if plot_pred:
            save_plot = input("Save the plot to a file? (y/n): ").strip().lower() == 'y'
            plot_output = None
            if save_plot:
                plot_output = input("Enter plot file path (leave blank for default): ").strip()

            plot_predictions(
                predictions,
                show_plot=True,
                save_plot=save_plot,
                output_file=plot_output if plot_output else None
            )

    return predictions


def predict_stock(ticker=config.DEFAULT_TICKER, days_back=30, model_name=None,
                  prediction_days=config.PREDICTION_HORIZON, model_dir=config.DEFAULT_MODELS_DIR):
    """
    Make stock price predictions

    Args:
        ticker (str): Stock ticker symbol
        days_back (int): Number of days of historical data to use
        model_name (str): Name of the model to use (None for best model)
        prediction_days (int): Number of days ahead to predict
        model_dir (str): Directory with trained models

    Returns:
        dict: Dictionary with prediction results
    """
    # Set date range for historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    print(f"\nLoading historical data for {ticker} from {start_date.date()} to {end_date.date()}")

    # Load data
    data_loader = DataLoader()
    merged_data = data_loader.merge_data_for_ml(ticker, start_date, end_date)

    if merged_data.empty:
        print("Error: No data available for the specified parameters.")
        return None

    print(f"Loaded {len(merged_data)} rows of data")

    # Preprocess data
    preprocessor = DataPreprocessor(use_scaling=config.USE_SCALING, scale_type=config.SCALING_TYPE)

    # First prepare features to include all lag features
    processed_data = preprocessor.prepare_features(merged_data)

    # Get features and target
    X, _ = preprocessor.get_feature_target_split(processed_data)

    # Make sure to fit the scaler after all features are created
    if config.USE_SCALING:
        preprocessor.fit_scaler(X)

    # Get the latest data for prediction
    latest_data = processed_data.iloc[-1:].copy()

    # Load trained models
    predictor = StockPredictionModels(model_dir=model_dir)

    # If no specific model is provided, find the best model
    if model_name is None:
        # First check if evaluation results exist
        results_file = os.path.join(model_dir, f"{ticker}_evaluation_results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    model_name = results.get('best_model')
                    if model_name:
                        print(f"Using best model from evaluation: {model_name}")
                    else:
                        model_name = "linear"  # Default to linear if best_model not found
                        print(f"No best model found in results. Defaulting to {model_name} model.")
            except Exception as e:
                model_name = "linear"  # Default to linear if there's an error reading the file
                print(f"Error reading evaluation results: {e}. Defaulting to {model_name} model.")
        else:
            # If no results file, just use a simple model that's likely to exist
            model_name = "linear"
            print(f"No evaluation results found. Defaulting to {model_name} model.")

    # Check if the model file exists before trying to load it
    model_file = os.path.join(model_dir, f"{model_name}_model.joblib")
    if not os.path.exists(model_file):
        print(f"Model file {model_file} not found.")

        # Try to find any available model file
        available_models = []
        for file in os.listdir(model_dir):
            if file.endswith("_model.joblib"):
                available_models.append(file.replace("_model.joblib", ""))

        if available_models:
            model_name = available_models[0]  # Use the first available model
            print(f"Using available model instead: {model_name}")
        else:
            print("No trained models found. Please train models first using the train.py script.")
            return None

    # Load the selected model
    try:
        model = predictor.load_model(model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Please train the model first using the train.py script.")
        return None

    print(f"Using {model_name} model for predictions")

    # Prepare data for multi-day predictions
    predictions = {}
    current_data = latest_data.copy()

    # Get feature column names
    X, _ = preprocessor.get_feature_target_split(processed_data)
    feature_cols = X.columns.tolist()

    # Make predictions for multiple days ahead
    for day in range(1, prediction_days + 1):
        print(f"Predicting for day {day}...")

        # Extract features for prediction
        X_pred = current_data[feature_cols].iloc[-1:].copy()

        # If scaling was used, apply it to prediction data
        if config.USE_SCALING and preprocessor.scaler:
            # No need to fit again, just transform
            X_pred_scaled = pd.DataFrame(
                preprocessor.scaler.transform(X_pred),
                columns=X_pred.columns,
                index=X_pred.index
            )
            # Use the scaled features for prediction
            predicted_return = predictor.predict(model_name, X_pred_scaled)[0]
        else:
            # Use the original features for prediction
            predicted_return = predictor.predict(model_name, X_pred)[0]

        # Calculate predicted price based on the last close price
        last_close = current_data['Close'].iloc[-1]
        predicted_price = last_close * (1 + predicted_return)

        # Store prediction
        prediction_date = (end_date + timedelta(days=day)).strftime('%Y-%m-%d')
        predictions[prediction_date] = {
            'predicted_return': float(predicted_return),
            'predicted_price': float(predicted_price)
        }

        # Update current data for next day prediction
        new_row = current_data.iloc[-1:].copy()
        new_row.index = [new_row.index[0] + 1]

        # Update date
        if 'Date' in new_row.columns:
            new_row['Date'] = end_date + timedelta(days=day)
        if 'date' in new_row.columns:
            new_row['date'] = end_date + timedelta(days=day)

        # Update Close price with prediction
        new_row['Close'] = predicted_price

        # Update Daily_Return with prediction
        new_row['Daily_Return'] = predicted_return

        # Shift features for next prediction
        for lag in range(5, 0, -1):
            if f'close_lag_{lag}' in new_row.columns:
                if lag > 1:
                    new_row[f'close_lag_{lag}'] = current_data[f'close_lag_{lag - 1}'].iloc[-1]
                else:
                    new_row[f'close_lag_1'] = current_data['Close'].iloc[-1]

        # Append to current data
        current_data = pd.concat([current_data, new_row])

    # Create summary
    summary = {
        'ticker': ticker,
        'prediction_date': datetime.now().strftime('%Y-%m-%d'),
        'model_used': model_name,
        'predictions': predictions
    }

    # Print predictions
    print("\nPrediction Summary:")
    print(f"Ticker: {ticker}")
    print(f"Model: {model_name}")
    print("\nPredicted Prices:")
    for date, pred in predictions.items():
        print(f"{date}: ${pred['predicted_price']:.2f} (Return: {pred['predicted_return'] * 100:.2f}%)")

    return summary


def save_predictions(predictions, output_file=None):
    """
    Save predictions to a JSON file

    Args:
        predictions (dict): Prediction results
        output_file (str): Path to output file (None for default)

    Returns:
        str: Path to saved file
    """
    if output_file is None:
        ticker = predictions['ticker']
        date_str = datetime.now().strftime('%Y%m%d')
        output_file = f"predictions/{ticker}_predictions_{date_str}.json"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save predictions to file
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=4)

    print(f"Predictions saved to {output_file}")
    return output_file


def plot_predictions(predictions, show_plot=True, save_plot=False, output_file=None):
    """
    Plot predictions

    Args:
        predictions (dict): Prediction results
        show_plot (bool): Whether to show the plot
        save_plot (bool): Whether to save the plot
        output_file (str): Path to save the plot (None for default)

    Returns:
        None
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("Error: matplotlib is required for plotting. Install with 'pip install matplotlib'")
        return

    # Extract data for plotting
    dates = list(predictions['predictions'].keys())
    prices = [pred['predicted_price'] for pred in predictions['predictions'].values()]
    returns = [pred['predicted_return'] * 100 for pred in predictions['predictions'].values()]

    # Convert dates to datetime
    dates_dt = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot predicted prices
    ax1.plot(dates_dt, prices, 'o-', color='blue', label='Predicted Price')
    ax1.set_title(f"{predictions['ticker']} Price Prediction")
    ax1.set_ylabel('Price ($)')
    ax1.grid(True)
    ax1.legend()

    # Format y-axis to show dollar amounts
    ax1.yaxis.set_major_formatter('${x:.2f}')

    # Plot predicted returns
    colors = ['green' if r > 0 else 'red' for r in returns]
    ax2.bar(dates_dt, returns, color=colors)
    ax2.set_title(f"{predictions['ticker']} Predicted Daily Returns")
    ax2.set_ylabel('Return (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True)

    # Format x-axis to show dates nicely
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    # Format y-axis to show percentages
    ax2.yaxis.set_major_formatter('{x:.2f}%')

    # Add a horizontal line at y=0 for returns
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save plot if requested
    if save_plot:
        if output_file is None:
            ticker = predictions['ticker']
            date_str = datetime.now().strftime('%Y%m%d')
            output_file = f"predictions/{ticker}_prediction_plot_{date_str}.png"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")

    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def add_fix_to_preprocessor():
    """
    Instructions to fix the DataPreprocessor class by adding a fit_scaler method
    """
    print("\n=== Instructions to Fix DataPreprocessor Class ===")
    print("You need to add a fit_scaler method to your DataPreprocessor class.")
    print("Open preprocessing.py and add the following method to the DataPreprocessor class:")

    fix_code = """
    def fit_scaler(self, data):
        \"\"\"
        Fit the scaler to the data

        Args:
            data (pd.DataFrame): Data to fit the scaler to

        Returns:
            None
        \"\"\"
        if self.use_scaling:
            if self.scale_type == 'standard':
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
            elif self.scale_type == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                self.scaler = MinMaxScaler()
            elif self.scale_type == 'robust':
                from sklearn.preprocessing import RobustScaler
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling type: {self.scale_type}")

            self.scaler.fit(data)
    """
    print(fix_code)
    print("\nThen you can use the interactive script to predict stock prices.")


if __name__ == "__main__":
    # Show information about the DataPreprocessor fix
    add_fix_to_preprocessor()

    print("\nWould you like to run the interactive stock prediction?")
    choice = input("Enter 'y' to continue, any other key to exit: ").strip().lower()

    if choice == 'y':
        # Run the interactive prediction
        predictions = interactive_predict_stock()