"""
Data loader module to retrieve data from MongoDB for machine learning pipeline.
"""
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from pymongo import MongoClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DataLoader:
    def __init__(self, db_uri=config.MONGO_URI):
        """Initialize the data loader with MongoDB connection"""
        self.client = MongoClient(db_uri)
        self.db = self.client[config.DB_NAME]

    def load_stock_data(self, ticker="AAPL", start_date=None, end_date=None):
        """Load stock data from MongoDB"""
        query = {"ticker": ticker}

        if start_date:
            if not end_date:
                end_date = datetime.now()

            # Convert dates to strings if they aren't already
            if isinstance(start_date, datetime):
                start_date = start_date.strftime("%Y-%m-%dT%H:%M:%S")
            if isinstance(end_date, datetime):
                end_date = end_date.strftime("%Y-%m-%dT%H:%M:%S")

            query["Date"] = {
                "$gte": start_date,
                "$lte": end_date
            }

        # Get data from stock collection
        stock_data = self.db.stock.find(query)
        df_stock = pd.DataFrame(list(stock_data))

        if not df_stock.empty:
            # Convert Date to datetime if it's string
            if isinstance(df_stock["Date"].iloc[0], str):
                df_stock["Date"] = pd.to_datetime(df_stock["Date"])

            # Sort by date
            df_stock = df_stock.sort_values(by="Date")

            # Calculate daily returns if they don't exist
            if "Daily_Return" not in df_stock.columns or df_stock["Daily_Return"].isnull().all():
                df_stock["Daily_Return"] = df_stock["Close"].pct_change()

        return df_stock

    def load_sentiment_data(self, start_date=None, end_date=None):
        """Load and aggregate sentiment data from news, Reddit, and Twitter"""
        # Create date range if provided
        news_query = {}
        reddit_query = {}
        twitter_query = {}

        if start_date:
            if not end_date:
                end_date = datetime.now()

            # Convert dates to strings if they aren't already
            if isinstance(start_date, datetime):
                start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                start_date_str = start_date

            if isinstance(end_date, datetime):
                end_date_str = end_date.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                end_date_str = end_date

            news_query["publishedAt"] = {
                "$gte": start_date_str,
                "$lte": end_date_str
            }

            reddit_query["created_at"] = {
                "$gte": start_date_str,
                "$lte": end_date_str
            }

            twitter_query["created_at"] = {
                "$gte": start_date_str,
                "$lte": end_date_str
            }

        # Load news data
        news_data = self.db.news.find(news_query)
        df_news = pd.DataFrame(list(news_data))

        # Load Reddit data
        reddit_data = self.db.reddit.find(reddit_query)
        df_reddit = pd.DataFrame(list(reddit_data))

        # Load Twitter data
        twitter_data = self.db.twitter.find(twitter_query)
        df_twitter = pd.DataFrame(list(twitter_data))

        # Process and prepare all sentiment data
        sentiment_dfs = []

        # Process news sentiment
        if not df_news.empty:
            # Convert date
            df_news["date"] = pd.to_datetime(df_news["publishedAt"]).dt.date
            news_agg = df_news.groupby("date").agg({
                "title_compound": "mean",
                "sentiment": lambda x: x.value_counts().index[0] if not x.empty else "neutral"
            }).reset_index()
            news_agg["source"] = "news"
            sentiment_dfs.append(news_agg)

        # Process Reddit sentiment
        if not df_reddit.empty:
            # Convert date
            df_reddit["date"] = pd.to_datetime(df_reddit["created_at"]).dt.date
            reddit_agg = df_reddit.groupby("date").agg({
                "title_compound": "mean",
                "sentiment": lambda x: x.value_counts().index[0] if not x.empty else "neutral"
            }).reset_index()
            reddit_agg["source"] = "reddit"
            sentiment_dfs.append(reddit_agg)

        # Process Twitter sentiment
        if not df_twitter.empty:
            # Convert date
            df_twitter["date"] = pd.to_datetime(df_twitter["created_at"]).dt.date
            twitter_agg = df_twitter.groupby("date").agg({
                "text_compound": "mean",
                "sentiment": lambda x: x.value_counts().index[0] if not x.empty else "neutral"
            }).reset_index()
            twitter_agg.rename(columns={"text_compound": "title_compound"}, inplace=True)
            twitter_agg["source"] = "twitter"
            sentiment_dfs.append(twitter_agg)

        # Combine all sentiment data
        if sentiment_dfs:
            combined_sentiment = pd.concat(sentiment_dfs, ignore_index=True)

            # Calculate daily average sentiment across all sources
            daily_sentiment = combined_sentiment.groupby("date").agg({
                "title_compound": "mean",
                "sentiment": lambda x: x.value_counts().index[0] if not x.empty else "neutral"
            }).reset_index()

            daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"])
            return daily_sentiment

        return pd.DataFrame()

    def load_economic_data(self, indicator="GDP", country="US", start_date=None, end_date=None):
        """Load economic indicator data"""
        query = {"indicator": indicator, "country": country}

        if start_date:
            if not end_date:
                end_date = datetime.now()

            # Convert dates to strings if they aren't already
            if isinstance(start_date, datetime):
                start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                start_date_str = start_date

            if isinstance(end_date, datetime):
                end_date_str = end_date.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                end_date_str = end_date

            query["date"] = {
                "$gte": start_date_str,
                "$lte": end_date_str
            }

        # Get data from economic collection
        economic_data = self.db.economic.find(query)
        df_economic = pd.DataFrame(list(economic_data))

        if not df_economic.empty:
            # Convert date to datetime if it's string
            if isinstance(df_economic["date"].iloc[0], str):
                df_economic["date"] = pd.to_datetime(df_economic["date"])

            # Sort by date
            df_economic = df_economic.sort_values(by="date")

        return df_economic

    def merge_data_for_ml(self, ticker="AAPL", start_date=None, end_date=None):
        """Merge stock, sentiment, and economic data for machine learning"""
        # Load stock data
        df_stock = self.load_stock_data(ticker, start_date, end_date)

        if df_stock.empty:
            print(f"No stock data found for ticker {ticker}")
            return pd.DataFrame()

        # Convert stock date to date only for joining
        df_stock = df_stock.copy()  # Create explicit copy to avoid SettingWithCopyWarning
        df_stock["date"] = pd.to_datetime(df_stock["Date"]).dt.date
        df_stock["date"] = pd.to_datetime(df_stock["date"])

        # Load sentiment data
        df_sentiment = self.load_sentiment_data(start_date, end_date)

        if df_sentiment.empty:
            print("No sentiment data found")
            # Add empty sentiment columns
            df_stock["title_compound"] = None
            df_stock["sentiment"] = "neutral"
            df_stock["sentiment_numeric"] = 0
            df_stock["target_next_day_return"] = df_stock["Daily_Return"].shift(-1)
            return df_stock

        # Merge stock data with sentiment data
        merged_df = pd.merge(df_stock, df_sentiment, on="date", how="left")

        # Use proper pandas methods to fill missing data (avoiding the deprecated warnings)
        merged_df = merged_df.copy()  # Create explicit copy to avoid SettingWithCopyWarning
        # Fill missing sentiment data
        merged_df["title_compound"] = merged_df["title_compound"].ffill()
        merged_df["sentiment"] = merged_df["sentiment"].fillna("neutral")

        # Create numerical sentiment feature
        sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
        merged_df["sentiment_numeric"] = merged_df["sentiment"].map(sentiment_map)

        # Calculate target variable (next day's return)
        merged_df["target_next_day_return"] = merged_df["Daily_Return"].shift(-1)

        return merged_df

    def check_collection_structure(self):
        """Utility method to check the structure of collections in the database"""
        collections = self.db.list_collection_names()
        print(f"Available collections: {collections}")

        # Check a sample document from each collection
        for collection_name in collections:
            sample = self.db[collection_name].find_one()
            if sample:
                print(f"\nSample from {collection_name}:")
                print(f"Fields: {list(sample.keys())}")
            else:
                print(f"\nCollection {collection_name} is empty")

        # Check available tickers
        if "stock" in collections:
            tickers = list(self.db.stock.distinct("ticker"))
            print(f"\nAvailable tickers: {tickers}")

            # Check record count for each ticker
            for ticker in tickers:
                count = self.db.stock.count_documents({"ticker": ticker})
                print(f"Ticker {ticker}: {count} records")

    def __del__(self):
        """Close MongoDB connection when object is deleted"""
        try:
            self.client.close()
        except Exception as e:
            # Handle potential errors during shutdown
            pass


if __name__ == "__main__":
    # Example usage
    data_loader = DataLoader()

    # First, let's check the database structure
    print("Checking database structure...")
    data_loader.check_collection_structure()

    # Example: Load data for known tickers
    print("\nLoading data for analysis...")

    # Try with MSFT ticker
    ticker = "MSFT"
    print(f"\nLoading data for {ticker}...")

    # Get data without date filtering first
    stock_data = data_loader.load_stock_data(ticker)
    print(f"Found {len(stock_data)} records for {ticker} (no date filter)")

    if not stock_data.empty:
        print(f"Date range: {stock_data['Date'].min()} to {stock_data['Date'].max()}")

        # For demonstration, let's use a smaller date range for faster processing
        # Use the last 90 days for faster processing
        end_date = stock_data['Date'].max()
        start_date = end_date - timedelta(days=90)
        print(f"\nLoading data with date range: {start_date} to {end_date}")

        # Load merged data
        merged_data = data_loader.merge_data_for_ml(ticker, start_date, end_date)
        print(f"Loaded {len(merged_data)} rows of merged data")

        if not merged_data.empty:
            print("\nData summary:")
            print(merged_data.head())

            # Add some more descriptive statistics
            print("\nDescriptive statistics:")
            print(merged_data[["Close", "Daily_Return", "title_compound", "sentiment_numeric"]].describe())

            # Count sentiment distribution
            if "sentiment" in merged_data.columns:
                print("\nSentiment distribution:")
                print(merged_data["sentiment"].value_counts())
        else:
            print("No data available for the specified parameters.")
    else:
        print(f"No data found for ticker {ticker}")