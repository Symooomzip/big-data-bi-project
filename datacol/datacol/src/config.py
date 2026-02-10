"""
Configuration file for stock prediction project
"""

# MongoDB configuration
MONGO_URI = "mongodb+srv://Lubabah:1234@cluster0.o5hydr7.mongodb.net/?retryWrites=true&w=majority&appName=cluster0"
DB_NAME = "predict_stock"

# Default parameters
DEFAULT_TICKER = "AAPL"
DEFAULT_DAYS_BACK = 365
DEFAULT_MODELS_DIR = "machine_learning/models"

# Training parameters
TRAIN_TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.25

# Model selection metric
BEST_MODEL_METRIC = "directional_accuracy"  # or "rmse" or "r2"

# Feature engineering settings
USE_SCALING = True
SCALING_TYPE = "standard"  # or "minmax"

# Prediction settings
PREDICTION_HORIZON = 5  # Number of days ahead to predict