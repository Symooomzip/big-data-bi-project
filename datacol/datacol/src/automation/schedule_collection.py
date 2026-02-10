import asyncio
import json
import os
import hashlib
import schedule
import time
import logging
import traceback

from src.data_collection.economic_scraper import collect_economic_data
from src.data_collection.twitter_scraper import collect_twitter_data
from src.data_collection.reddit_scraper import collect_reddit_data
from src.data_collection.stock_scraper import collect_stock_data
from src.data_collection.news_scraper import collect_news_data
from src.data_processing.clean_data import process_raw_data
from src.machine_learning.train import train_models
from src.mongodb.mongo_handler import MongoDBHandler

# Set up logging for scheduled tasks
log_file = 'scheduled_task_log.txt'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize MongoDB handler
mongo_handler = MongoDBHandler()

# Cache to track recent items to prevent duplicates
# This is a simple in-memory solution that will reset when the script restarts
recent_items_cache = {
    'stock': set(),
    'news': set(),
    'economic': set(),
    'twitter': set(),
    'reddit': set()
}

# Maximum cache size per data type to prevent memory issues
MAX_CACHE_SIZE = 10000


def generate_content_hash(item):
    """
    Generate a hash for an item to identify duplicates.
    This function creates a unique signature based on the content of the data.
    """
    try:
        if not isinstance(item, dict):
            return None

        # Create a sorted string representation of the item's key contents
        # This approach handles different field orders but same content
        content_fields = []

        # Common unique identifiers to check
        key_fields = ['id', 'title', 'url', 'text', 'content', 'tweet_id', 'post_id', 'symbol', 'timestamp']

        for field in key_fields:
            if field in item and item[field]:
                content_fields.append(f"{field}:{str(item[field])}")

        # If no identifying fields were found, use the whole item
        if not content_fields:
            content_str = json.dumps(item, sort_keys=True)
        else:
            content_str = "||".join(content_fields)

        # Generate a hash of the content string
        return hashlib.md5(content_str.encode()).hexdigest()
    except:
        # If anything goes wrong, return None
        return None


def deduplicate_data(data, data_type):
    """
    Remove duplicate items from the data.
    Returns a list of unique items.
    """
    if not data or not isinstance(data, list):
        return data

    unique_items = []
    unique_hashes = set()

    # Get the cache for this data type
    cache = recent_items_cache.get(data_type, set())

    # Track how many duplicates we find
    duplicate_count = 0

    for item in data:
        item_hash = generate_content_hash(item)

        # Skip items without a valid hash
        if not item_hash:
            unique_items.append(item)
            continue

        # Check if this item is a duplicate
        if item_hash not in cache and item_hash not in unique_hashes:
            unique_items.append(item)
            unique_hashes.add(item_hash)
        else:
            duplicate_count += 1

    # Update the cache with new items
    cache.update(unique_hashes)

    # If cache is too large, remove oldest items (this is a simplification)
    if len(cache) > MAX_CACHE_SIZE:
        # Convert to list to use pop
        cache_list = list(cache)
        # Remove oldest items (first in the list)
        cache = set(cache_list[len(cache_list) - MAX_CACHE_SIZE:])

    # Update the global cache
    recent_items_cache[data_type] = cache

    logging.info(f"Deduplication: {duplicate_count} duplicates removed from {data_type} data")

    return unique_items


# Define a function to run all collection tasks, clean the data, and then save
def collect_and_process_all_data():
    try:
        # Collect raw data
        logging.info("Starting data collection process...")

        # Stock data collection
        logging.info("Collecting Stock Data...")
        stock_data = collect_stock_data()
        if stock_data:
            logging.info(
                f"Successfully collected {len(stock_data) if isinstance(stock_data, list) else 'unknown'} stock data records")
        else:
            logging.warning("Stock data collection returned empty or None")

        # News data collection - add more debugging
        logging.info("Collecting News Data...")
        news_data = collect_news_data()
        if news_data:
            logging.info(
                f"Successfully collected {len(news_data) if isinstance(news_data, list) else 'unknown'} news data records")
            # Log a sample of the first news item if available
            if isinstance(news_data, list) and len(news_data) > 0:
                logging.info(
                    f"Sample news item keys: {list(news_data[0].keys()) if isinstance(news_data[0], dict) else 'not a dict'}")
        else:
            logging.warning("News data collection returned empty or None - check your news_scraper.py implementation")
            # Add a hint about what might be wrong
            logging.info("Hint: Check your news API key, URL endpoints, and response handling in collect_news_data()")

        # Economic data collection
        logging.info("Collecting Economic Data...")
        economic_data = collect_economic_data()
        if economic_data:
            logging.info(
                f"Successfully collected {len(economic_data) if isinstance(economic_data, list) else 'unknown'} economic data records")
        else:
            logging.warning("Economic data collection returned empty or None")

        # Twitter data collection with improved logging
        logging.info("Collecting Twitter Data...")
        twitter_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(twitter_loop)
        try:
            twitter_data = twitter_loop.run_until_complete(collect_twitter_data())

            # Add validation to check if twitter_data is valid
            if twitter_data and isinstance(twitter_data, list) and len(twitter_data) > 0:
                logging.info(f"Successfully collected {len(twitter_data)} Twitter records")

                # Save the collected raw data to a file to ensure it's available for processing
                # This ensures the data is available for the cleaning step even if there's an issue with passing it
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Adjust if needed
                raw_data_path = os.path.join(project_root, 'data', 'raw', 'twitter_data.json')

                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)

                # Save the data again to ensure it's properly written
                with open(raw_data_path, 'w') as f:
                    json.dump(twitter_data, f, indent=4)

                logging.info(f"Twitter data saved to {raw_data_path}")

                # Print a sample tweet for debugging
                if len(twitter_data) > 0:
                    logging.info(f"Sample Twitter data: {twitter_data[0]}")
            else:
                logging.warning(f"Twitter data collection returned empty, None, or invalid data: {type(twitter_data)}")
                if twitter_data is None:
                    logging.warning("Twitter data is None")
                elif isinstance(twitter_data, list) and len(twitter_data) == 0:
                    logging.warning("Twitter data is an empty list")
                else:
                    logging.warning(f"Twitter data is of unexpected type: {type(twitter_data)}")
        except Exception as e:
            logging.error(f"Error during Twitter data collection: {e}")
            logging.error(traceback.format_exc())
            twitter_data = []  # Initialize to empty list on error
        finally:
            twitter_loop.close()

        # Reddit data collection
        logging.info("Collecting Reddit Data...")
        reddit_data = collect_reddit_data()
        if reddit_data:
            logging.info(
                f"Successfully collected {len(reddit_data) if isinstance(reddit_data, list) else 'unknown'} Reddit records")
        else:
            logging.warning("Reddit data collection returned empty or None")

        logging.info("All data collection complete")

        # Process raw data with enhanced error handling
        logging.info("Starting to process and clean the collected data...")

        # Process each data type with better error logging
        data_types = ['stock', 'news', 'economic', 'twitter', 'reddit']
        cleaned_data = {}

        for data_type in data_types:
            try:
                logging.info(f"Processing {data_type} data...")
                cleaned_result = process_raw_data(data_type)

                # Deduplicate the data before storing
                cleaned_result = deduplicate_data(cleaned_result, data_type)

                # Store the cleaned data
                cleaned_data[data_type] = cleaned_result

                # Log the result
                if cleaned_result:
                    logging.info(f"Cleaned {data_type} data: {len(cleaned_result)} records after deduplication")
                else:
                    logging.warning(f"Cleaned {data_type} data is empty after processing")
            except Exception as e:
                logging.error(f"Error processing {data_type} data: {e}")
                logging.error(traceback.format_exc())
                cleaned_data[data_type] = []

        # Insert cleaned data into MongoDB with better error handling
        for data_type, data in cleaned_data.items():
            if data:
                try:
                    # Use upsert for data types where duplicates should be updated
                    if data_type in ['stock', 'economic']:
                        # These types typically have defined identifiers that should be updated
                        mongo_handler.upsert_data(data, data_type)
                        logging.info(
                            f"{data_type.capitalize()} data upserted into MongoDB {data_type} collection: {len(data)} records")
                    else:
                        # For social media and news, just insert unique items
                        # (We've already deduplicated these in memory)
                        mongo_handler.insert_data(data, data_type)
                        logging.info(
                            f"{data_type.capitalize()} data inserted into MongoDB {data_type} collection: {len(data)} records")
                except Exception as e:
                    logging.error(f"Failed to insert {data_type} data into MongoDB: {e}")
                    logging.error(traceback.format_exc())
            else:
                logging.info(f"No {data_type} data to insert into MongoDB")

        logging.info("All data processed and inserted into MongoDB collections.")

        # Call the train_models function to retrain the stock prediction models for all tickers
        tickers = ['AAPL', 'SPY', 'GOOGL', 'MSFT']
        for ticker in tickers:
            logging.info(f"Starting to retrain the model for ticker {ticker}...")
            train_models(ticker=ticker, days_back=60)  # Customize days_back as needed
            logging.info(f"Completed retraining for ticker {ticker}")

    except Exception as e:
        logging.error(f"Error during data collection or processing: {e}")
        logging.error(traceback.format_exc())


# If you want to run immediately on startup
collect_and_process_all_data()

# Schedule collection and cleaning every 15 minutes
schedule.every(15).minutes.do(collect_and_process_all_data)

# Keep running the schedule
while True:
    schedule.run_pending()
    time.sleep(1)