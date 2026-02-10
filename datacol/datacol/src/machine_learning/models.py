import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Fix the XGBoost import
from xgboost import XGBRegressor
import joblib
import os


class StockPredictionModels:
    def __init__(self, model_dir='models'):
        """
        Initialize the stock prediction models

        Args:
            model_dir (str): Directory to save trained models
        """
        self.model_dir = model_dir
        self.models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': XGBRegressor(n_estimators=100, random_state=42)  # Now correctly instantiated
        }
        self.trained_models = {}

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

    def train_model(self, model_name, X_train, y_train):
        """
        Train a specific model

        Args:
            model_name (str): Name of the model to train
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target

        Returns:
            object: Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

        # Get the model
        model = self.models[model_name]

        # Train the model
        model.fit(X_train, y_train)

        # Store the trained model
        self.trained_models[model_name] = model

        return model

    def train_all_models(self, X_train, y_train):
        """
        Train all available models

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target

        Returns:
            dict: Dictionary of trained models
        """
        for model_name in self.models:
            self.train_model(model_name, X_train, y_train)

        return self.trained_models

    def evaluate_model(self, model_name, X_val, y_val):
        """
        Evaluate a specific model

        Args:
            model_name (str): Name of the model to evaluate
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target

        Returns:
            dict: Dictionary of evaluation metrics
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet. Train it first.")

        # Get the trained model
        model = self.trained_models[model_name]

        # Make predictions
        y_pred = model.predict(X_val)

        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'mae': mean_absolute_error(y_val, y_pred),
            'r2': r2_score(y_val, y_pred)
        }

        # Calculate directional accuracy (positive vs negative)
        direction_correct = np.sum((y_val > 0) == (y_pred > 0)) / len(y_val)
        metrics['directional_accuracy'] = direction_correct

        return metrics

    def evaluate_all_models(self, X_val, y_val):
        """
        Evaluate all trained models

        Args:
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target

        Returns:
            dict: Dictionary of evaluation metrics for all models
        """
        results = {}

        for model_name in self.trained_models:
            results[model_name] = self.evaluate_model(model_name, X_val, y_val)

        # Create a DataFrame with results
        results_df = pd.DataFrame(results).T

        return results_df

    def select_best_model(self, results_df, metric='directional_accuracy'):
        """
        Select the best model based on a specific metric

        Args:
            results_df (pd.DataFrame): DataFrame with evaluation results
            metric (str): Metric to use for selection

        Returns:
            str: Name of the best model
        """
        if metric in ['rmse', 'mae']:
            # Lower is better for these metrics
            best_model = results_df[metric].idxmin()
        else:
            # Higher is better for r2, directional_accuracy
            best_model = results_df[metric].idxmax()

        return best_model

    def predict(self, model_name, X):
        """
        Make predictions with a specific model

        Args:
            model_name (str): Name of the model to use
            X (pd.DataFrame): Features for prediction

        Returns:
            np.array: Predicted values
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet. Train it first.")

        # Get the trained model
        model = self.trained_models[model_name]

        # Make predictions
        predictions = model.predict(X)

        return predictions

    def feature_importance(self, model_name):
        """
        Get feature importance for a specific model

        Args:
            model_name (str): Name of the model

        Returns:
            pd.DataFrame: DataFrame with feature importance
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet. Train it first.")

        # Get the trained model
        model = self.trained_models[model_name]

        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            return None

        # Get feature importances and feature names
        feature_names = self.trained_models[model_name].feature_names_in_
        importances = model.feature_importances_

        # Create DataFrame with feature importances
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df

    def save_model(self, model_name):
        """
        Save a trained model to disk

        Args:
            model_name (str): Name of the model to save

        Returns:
            str: Path to the saved model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet. Train it first.")

        # Create file path
        file_path = os.path.join(self.model_dir, f"{model_name}_model.joblib")

        # Save the model
        joblib.dump(self.trained_models[model_name], file_path)

        return file_path

    def load_model(self, model_name):
        """
        Load a trained model from disk

        Args:
            model_name (str): Name of the model to load

        Returns:
            object: Loaded model
        """
        # Create file path
        file_path = os.path.join(self.model_dir, f"{model_name}_model.joblib")

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file {file_path} not found.")

        # Load the model
        model = joblib.load(file_path)

        # Store the loaded model
        self.trained_models[model_name] = model

        return model