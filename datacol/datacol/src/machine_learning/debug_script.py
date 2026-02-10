"""
Script to debug MongoDB connection and data retrieval
"""
import pymongo
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

# Assuming config.py is in the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def debug_mongo_connection():
    """Test MongoDB connection and explore data structure"""
    print("Testing MongoDB connection...")

    # Connect to MongoDB
    client = pymongo.MongoClient(config.MONGODB_URI)
    db = client[config.DB_NAME]

    # Print available collections
    collections = db.list_collection_names()
    print(f"Available collections: {collections}")

    # Check ticker format in stock collection
    if "stock" in collections:
        # Get all unique tickers
        tickers = list(db.stock.distinct("ticker"))
        print(f"\nAvailable tickers: {tickers[:10]}{'...' if len(tickers) > 10 else ''}")

        # Try to retrieve AAPL data with different query formats
        queries = [
            {"ticker": "AAPL"},
            {"ticker": '"AAPL"'},
            {"ticker": {"$regex": "AAPL"}},
        ]

        for i, query in enumerate(queries):
            print(f"\nQuery {i + 1}: {query}")
            cursor = db.stock.find(query).limit(5)
            results = list(cursor)
            print(f"Found {len(results)} records")

            if results:
                sample = results[0]
                print("Sample fields:", list(sample.keys()))
                print(f"Ticker: {sample.get('ticker')}")
                print(f"Date: {sample.get('Date')}")

    # Check date fields format in all collections
    for collection_name in collections:
        sample = db[collection_name].find_one()
        if sample:
            print(f"\nCollection: {collection_name}")
            date_fields = [field for field in sample.keys()
                           if field.lower() in ["date", "created_at", "publishedat"]]

            for date_field in date_fields:
                date_value = sample[date_field]
                print(f"Field: {date_field}, Value: {date_value}, Type: {type(date_value)}")

                # Try to parse string dates
                if isinstance(date_value, str):
                    try:
                        parsed_date = pd.to_datetime(date_value)
                        print(f"  Parsed date: {parsed_date}")
                    except:
                        print("  Failed to parse date string")

    # Clean up
    client.close()
    print("\nConnection closed")


if __name__ == "__main__":
    debug_mongo_connection()