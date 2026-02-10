import json
import os
import twikit
import asyncio
import logging
from datetime import datetime


# Function to collect tweets for a given keyword
async def collect_keyword_tweets(client, term):
    logging.info(f"Searching for tweets related to: {term}")
    try:
        # Fetch tweets for a specific keyword
        tweets = await client.search_tweet(term, 'Latest')
        logging.info(f"Found {len(tweets) if tweets else 0} tweets for '{term}'")
        return tweets
    except twikit.errors.NotFound as e:
        logging.error(f"Error: Not found when searching for '{term}' - {e}")
        return []  # Return an empty list in case of error
    except Exception as e:
        logging.error(f"Error when searching for '{term}': {e}")
        return []  # Return an empty list in case of other errors


async def collect_twitter_data():
    logging.info("Starting Twitter data collection...")

    # Get the project root and config path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, 'config', 'twitter_config.json')

    # Check if the config file exists
    if not os.path.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load Twitter credentials
    try:
        with open(config_path, 'r') as f:
            twitter_config = json.load(f)
        logging.info("Twitter config loaded successfully")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing Twitter config file: {e}")
        return []
    except Exception as e:
        logging.error(f"Error loading Twitter config: {e}")
        return []

    try:
        # Initialize Twikit Client
        client = twikit.Client('en-US')

        # Login with credentials and set the Bearer Token
        await client.login(
            auth_info_1=twitter_config['TWITTER_USERNAME'],
            auth_info_2=twitter_config['TWITTER_EMAIL'],
            password=twitter_config['TWITTER_PASSWORD'],
            cookies_file='cookies.json'
        )
        logging.info("Successfully logged in to Twitter")

    except Exception as e:
        logging.error(f"Failed to initialize Twitter client or login: {e}")
        return []

    # List of keywords to search for
    event_keywords = ['global economic crisis', 'natural disaster', 'political unrest', 'inflation trends']
    logging.info(f"Searching for keywords: {event_keywords}")

    try:
        # Collect tweets for each keyword concurrently using asyncio.gather
        tasks = [collect_keyword_tweets(client, term) for term in event_keywords]
        results = await asyncio.gather(*tasks)

        # Check if any results were found
        total_tweets = sum(len(tweets) if tweets else 0 for tweets in results)
        logging.info(f"Total tweets found across all keywords: {total_tweets}")

        if total_tweets == 0:
            logging.warning("No tweets found for any keywords")
            # Return an empty list but don't terminate
            return []

    except Exception as e:
        logging.error(f"Error during tweet collection: {e}")
        return []

    # Process the results
    all_tweets_data = []

    for term, tweets in zip(event_keywords, results):
        if tweets:
            logging.info(f"Processing {len(tweets)} tweets for '{term}'")
            for tweet in tweets:
                try:
                    # Use a consistent date format: "Day Month DD HH:MM:SS +0000 YYYY"
                    # This matches the format in your sample data: "Sun May 11 12:58:38 +0000 2025"
                    current_datetime = datetime.now()
                    formatted_datetime = current_datetime.strftime('%a %b %d %H:%M:%S +0000 %Y')

                    # Extract tweet information with careful attribute checking
                    data = {
                        'text': getattr(tweet, 'text', ''),
                        'created_at': formatted_datetime,  # Use this consistent format
                        'author_name': getattr(getattr(tweet, 'user', {}), 'name', 'Unknown'),
                        'author_username': getattr(getattr(tweet, 'user', {}), 'screen_name', 'Unknown'),
                        'lang': getattr(tweet, 'lang', 'unknown'),
                        'event_type': term,
                        'sentiment': None  # Placeholder for sentiment analysis
                    }
                    all_tweets_data.append(data)
                except Exception as e:
                    logging.error(f"Error processing tweet for '{term}': {e}")
                    continue

    # Check if we have any processed tweets
    if not all_tweets_data:
        logging.warning("No tweets were successfully processed")
        return []  # Return empty list if no tweets processed

    # Save collected data to JSON file
    try:
        output_path = os.path.join(project_root, 'data', 'raw', 'twitter_data.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(all_tweets_data, f, indent=4)

        logging.info(f"Successfully saved {len(all_tweets_data)} tweets to {output_path}")
    except Exception as e:
        logging.error(f"Error saving Twitter data to file: {e}")
        # Continue and try to return the data anyway

    # Log a sample tweet to verify data structure
    if all_tweets_data:
        logging.info(f"Sample tweet: {all_tweets_data[0]}")

    logging.info(f"Returning {len(all_tweets_data)} tweets")
    return all_tweets_data