"""
Training script for stock price prediction models.
"""
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import argparse
import json

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from .data_loader import DataLoader
from .preprocessing import DataPreprocessor
from .models import StockPredictionModels
import config


# Add these improvements to your train.py file, in the train_models function:

def train_models(ticker=config.DEFAULT_TICKER, days_back=config.DEFAULT_DAYS_BACK,
                 model_names=None, save_models=True, output_dir=config.DEFAULT_MODELS_DIR):
    """
    Train stock prediction models

    Args:
        ticker (str): Stock ticker symbol
        days_back (int): Number of days to look back for training data
        model_names (list): List of model names to train (None for all)
        save_models (bool): Whether to save trained models
        output_dir (str): Directory to save trained models

    Returns:
        dict: Dictionary with training results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    print(f"Training models for {ticker} from {start_date.date()} to {end_date.date()}")

    # Load data
    print("Loading data...")
    data_loader = DataLoader()
    merged_data = data_loader.merge_data_for_ml(ticker, start_date, end_date)

    if merged_data.empty:
        print("Error: No data available for the specified parameters.")
        return None

    print(f"Loaded {len(merged_data)} rows of data")

    # Some checks for data quality
    missing_values = merged_data.isnull().sum().sum()
    if missing_values > 0:
        print(f"Warning: Found {missing_values} missing values in the data")

    # Preprocess data
    print("Preprocessing data...")
    try:
        preprocessor = DataPreprocessor(use_scaling=config.USE_SCALING, scale_type=config.SCALING_TYPE)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_pipeline(merged_data)

        # Check for empty data after preprocessing
        if X_train.empty or y_train.empty:
            print("Error: Preprocessing resulted in empty training data.")
            return None

        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

    # Train models
    print("Training models...")
    try:
        predictor = StockPredictionModels(model_dir=output_dir)

        # Choose which models to train
        available_models = list(predictor.models.keys())
        if model_names is None:
            # Train all available models
            models_to_train = available_models
        else:
            # Check if requested models exist
            models_to_train = [model for model in model_names if model in available_models]
            if not models_to_train:
                print(
                    f"Error: None of the requested models {model_names} are available. Available models: {available_models}")
                return None

        # Train specified models
        for model_name in models_to_train:
            try:
                print(f"Training {model_name} model...")
                predictor.train_model(model_name, X_train, y_train)
            except Exception as e:
                print(f"Error training {model_name} model: {e}")

        if not predictor.trained_models:
            print("Error: No models were successfully trained.")
            return None

    except Exception as e:
        print(f"Error during model training setup: {e}")
        return None

    # Continue with model evaluation and saving...
    # (rest of the function remains the same)

    # Evaluate models
    print("Evaluating models...")
    eval_results = predictor.evaluate_all_models(X_val, y_val)
    print("\nValidation Results:")
    print(eval_results)

    # Evaluate on test set
    test_results = {}
    for model_name in predictor.trained_models:
        y_test_pred = predictor.predict(model_name, X_test)
        test_rmse = ((y_test - y_test_pred) ** 2).mean() ** 0.5
        test_direction_correct = (y_test * y_test_pred > 0).mean()
        test_results[model_name] = {
            "test_rmse": test_rmse,
            "test_directional_accuracy": test_direction_correct
        }

    test_results_df = pd.DataFrame(test_results).T
    print("\nTest Results:")
    print(test_results_df)

    # Select best model
    best_model = predictor.select_best_model(eval_results, metric=config.BEST_MODEL_METRIC)
    print(f"\nBest model: {best_model}")

    # Get feature importance for the best model
    importance_df = predictor.feature_importance(best_model)
    if importance_df is not None:
        print("\nTop 10 Feature Importance:")
        print(importance_df.head(10))

    # Save models if requested
    if save_models:
        print("\nSaving models...")
        saved_models = {}
        for model_name in predictor.trained_models:
            model_path = predictor.save_model(model_name)
            saved_models[model_name] = model_path
            print(f"Saved {model_name} model to {model_path}")

    # Save evaluation results
    results_file = os.path.join(output_dir, f"{ticker}_evaluation_results.json")

    results = {
        "ticker": ticker,
        "train_start_date": start_date.strftime("%Y-%m-%d"),
        "train_end_date": end_date.strftime("%Y-%m-%d"),
        "validation_results": eval_results.to_dict(),
        "test_results": test_results_df.to_dict(),
        "best_model": best_model,
        "feature_importance": importance_df.to_dict() if importance_df is not None else None,
        "saved_models": saved_models if save_models else None
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to {results_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train stock prediction models')
    parser.add_argument('--ticker', type=str, default=config.DEFAULT_TICKER,
                        help='Stock ticker symbol')
    parser.add_argument('--days-back', type=int, default=config.DEFAULT_DAYS_BACK,
                        help='Number of days to look back for training data')
    parser.add_argument('--models', type=str, nargs='+',
                        help='List of model names to train (None for all)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save trained models')
    parser.add_argument('--output-dir', type=str, default=config.DEFAULT_MODELS_DIR,
                        help='Directory to save trained models')

    args = parser.parse_args()

    # Train models
    train_models(
        ticker=args.ticker,
        days_back=args.days_back,
        model_names=args.models,
        save_models=not args.no_save,
        output_dir=args.output_dir
    )