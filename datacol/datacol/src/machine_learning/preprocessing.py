"""
Preprocessing module for machine learning pipeline.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, use_scaling=True, scale_type="standard"):
        """
        Initialize the data preprocessor

        Args:
            use_scaling (bool): Whether to scale numerical features
            scale_type (str): Type of scaling to use ('standard' or 'minmax')
        """
        self.use_scaling = use_scaling
        self.scale_type = scale_type
        self.scaler = None

        if self.use_scaling:
            if self.scale_type == "standard":
                self.scaler = StandardScaler()
            elif self.scale_type == "minmax":
                self.scaler = MinMaxScaler()
            elif self.scale_type == "robust":
                self.scaler = RobustScaler()

    def fit_scaler(self, data):
        """
        Fit the scaler to the data

        Args:
            data (pd.DataFrame): Data to fit the scaler to

        Returns:
            None
        """
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

    def prepare_features(self, df):
        """
        Prepare features for machine learning

        Args:
            df (pd.DataFrame): DataFrame with merged stock and sentiment data

        Returns:
            pd.DataFrame: DataFrame with prepared features
        """
        # Make a copy to avoid modifying the original dataframe
        data = df.copy()

        # First, handle ticker column (convert to one-hot encoding if multiple tickers present)
        if 'ticker' in data.columns:
            unique_tickers = data['ticker'].nunique()
            if unique_tickers > 1:
                # For multiple tickers, create one-hot encoding
                ticker_dummies = pd.get_dummies(data['ticker'], prefix='ticker')
                data = pd.concat([data, ticker_dummies], axis=1)
            # Keep the original ticker column as it will be excluded in feature selection

        # Create lag features for stock data (past n days)
        for lag in range(1, 6):  # 1 to 5 day lags
            data[f'close_lag_{lag}'] = data['Close'].shift(lag)
            data[f'volume_lag_{lag}'] = data['Volume'].shift(lag)
            data[f'return_lag_{lag}'] = data['Daily_Return'].shift(lag)

        # Create lag features for sentiment data
        for lag in range(1, 4):  # 1 to 3 day lags
            if 'title_compound' in data.columns:
                data[f'sentiment_compound_lag_{lag}'] = data['title_compound'].shift(lag)
            if 'sentiment_numeric' in data.columns:
                data[f'sentiment_numeric_lag_{lag}'] = data['sentiment_numeric'].shift(lag)

        # Create rolling window features
        data['close_rolling_mean_3d'] = data['Close'].rolling(window=3).mean()
        data['close_rolling_mean_7d'] = data['Close'].rolling(window=7).mean()
        data['volume_rolling_mean_3d'] = data['Volume'].rolling(window=3).mean()

        # Volatility features (standard deviation of returns)
        data['return_rolling_std_5d'] = data['Daily_Return'].rolling(window=5).std()

        # Sentiment momentum (change in sentiment)
        if 'title_compound' in data.columns:
            data['sentiment_momentum'] = data['title_compound'] - data['title_compound'].shift(1)

        # Use volatility from original data if present
        if 'volatility' in data.columns:
            # Already exists, no need to recalculate
            pass
        else:
            # Calculate 5-day rolling volatility
            data['volatility'] = data['Daily_Return'].rolling(window=5).std()

        # Create day of week features (as numeric values instead of dummies)
        if 'Day_of_Week' not in data.columns and 'Date' in data.columns:
            data['Day_of_Week'] = pd.to_datetime(data['Date']).dt.dayofweek

        # Create month features (as numeric values instead of dummies)
        if 'Month' not in data.columns and 'Date' in data.columns:
            data['Month'] = pd.to_datetime(data['Date']).dt.month

        # Use sin and cos transformations for cyclical features instead of one-hot encoding
            # Use sin and cos transformations for cyclical features instead of one-hot encoding
            if 'Day_of_Week' in data.columns:
                # First ensure Day_of_Week is numeric
                if data['Day_of_Week'].dtype == 'object':
                    # Convert day names to integers (Monday=0, Tuesday=1, etc.)
                    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5,
                               'Sunday': 6}
                    data['Day_of_Week'] = data['Day_of_Week'].map(day_map).fillna(0).astype(int)

                # Convert day of week to cyclical features (0-6 to sin/cos)
                data['day_sin'] = np.sin(2 * np.pi * data['Day_of_Week'].astype(float) / 7)
                data['day_cos'] = np.cos(2 * np.pi * data['Day_of_Week'].astype(float) / 7)

            if 'Month' in data.columns:
                # First ensure Month is numeric
                if data['Month'].dtype == 'object':
                    # Convert month names to integers (January=1, February=2, etc.)
                    month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                                 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
                    data['Month'] = data['Month'].map(month_map).fillna(1).astype(int)

                # Convert month to cyclical features (1-12 to sin/cos)
                data['month_sin'] = np.sin(2 * np.pi * data['Month'].astype(float) / 12)
                data['month_cos'] = np.cos(2 * np.pi * data['Month'].astype(float) / 12)

        # Drop rows with NaN values created by lag features
        data = data.dropna()

        return data

    def get_feature_target_split(self, df, target_col='target_next_day_return'):
        """
        Split the data into features and target

        Args:
            df (pd.DataFrame): DataFrame with prepared features
            target_col (str): Name of the target column

        Returns:
            tuple: X (features) and y (target)
        """
        # Make a copy to avoid modifying the original dataframe
        df_copy = df.copy()

        # Define columns to exclude from features
        exclude_cols = [
            '_id', 'Date', 'date', 'sentiment', 'target_next_day_return',
            'Open', 'High', 'Low', 'Close', 'publishedAt', 'created_at',
            'ticker', 'Adj Close', 'Day_of_Week', 'Month',  # Added these columns
            # Add any other non-numeric or identifier columns here
        ]

        # Get feature columns (exclude non-feature columns and target)
        feature_cols = [col for col in df_copy.columns
                        if col not in exclude_cols and col != target_col]

        # Extract features and target
        X = df_copy[feature_cols].copy()

        # Print column dtypes for debugging
        print("Feature column data types before conversion:")
        print(X.dtypes)

        # Convert each column to numeric type and handle non-numeric columns
        numeric_cols = []
        for col in X.columns:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                numeric_cols.append(col)
            except Exception as e:
                print(f"Warning: Could not convert column '{col}' to numeric: {e}")

        # Keep only numeric columns
        X = X[numeric_cols]

        # Fill any NaN values created by the coercion
        X = X.fillna(0)

        # Print column dtypes after conversion for verification
        print("Feature column data types after conversion:")
        print(X.dtypes)

        # Make sure target is also numeric
        y = pd.to_numeric(df_copy[target_col], errors='coerce').fillna(0)

        return X, y

    def train_val_test_split(self, X, y, test_size=0.2, val_size=0.25):
        """
        Split data into train, validation, and test sets

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Proportion of data for testing
            val_size (float): Proportion of training data for validation

        Returns:
            tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Make sure X contains only numeric data
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) < len(X.columns):
            non_numeric = [col for col in X.columns if col not in numeric_cols]
            print(f"Warning: Dropping non-numeric columns: {non_numeric}")
            X = X[numeric_cols]

        # First split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # Time series data should not be shuffled
        )

        # Then split train+val into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, shuffle=False
        )

        # Scale features if needed
        if self.use_scaling and self.scaler:
            try:
                # Fit the scaler only on training data
                X_train_scaled = pd.DataFrame(
                    self.scaler.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index
                )

                # Transform validation and test sets using the fitted scaler
                X_val_scaled = pd.DataFrame(
                    self.scaler.transform(X_val),
                    columns=X_val.columns,
                    index=X_val.index
                )

                X_test_scaled = pd.DataFrame(
                    self.scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )

                return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
            except Exception as e:
                print(f"Error during scaling: {e}")
                print("Proceeding without scaling...")
                return X_train, X_val, X_test, y_train, y_val, y_test

        return X_train, X_val, X_test, y_train, y_val, y_test

    def handle_missing_values(self, df):
        """
        Handle missing values in the dataframe

        Args:
            df (pd.DataFrame): DataFrame with potentially missing values

        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        # Make a copy to avoid modifying the original
        data = df.copy()

        # First identify numeric columns
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

        # For numeric columns, fill NaNs with median
        for col in numeric_cols:
            if data[col].isnull().any():
                median_val = data[col].median()
                data[col] = data[col].fillna(median_val)
                print(f"Filled missing values in '{col}' with median: {median_val}")

        # For categorical columns, fill with mode
        cat_cols = data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if data[col].isnull().any():
                mode_val = data[col].mode()[0]
                data[col] = data[col].fillna(mode_val)
                print(f"Filled missing values in '{col}' with mode: {mode_val}")

        # Check if there are still any missing values
        missing = data.isnull().sum().sum()
        if missing > 0:
            print(f"Warning: There are still {missing} missing values after handling")
            # Drop rows with any remaining NaN values
            data = data.dropna()
            print(f"Dropped rows with NaN values, {len(data)} rows remaining")

        return data

    def preprocess_pipeline(self, df, target_col='target_next_day_return'):
        """
        Run the full preprocessing pipeline

        Args:
            df (pd.DataFrame): Raw merged dataframe
            target_col (str): Name of the target column

        Returns:
            tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First handle missing values
        print("Handling missing values...")
        clean_df = self.handle_missing_values(df)

        # Prepare features
        print("Preparing features...")
        processed_df = self.prepare_features(clean_df)

        # Split into features and target
        print("Splitting into features and target...")
        X, y = self.get_feature_target_split(processed_df, target_col)

        # Split into train, val, test sets
        print("Splitting into train, validation, and test sets...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.train_val_test_split(X, y)

        return X_train, X_val, X_test, y_train, y_val, y_test