import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='data_processing.log'
)


def get_project_root():
    """Helper function to get the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))


def save_cleaned_data(df, data_type):
    """ Save cleaned data to the respective directories. """
    if df is None or df.empty:
        logging.warning(f"No cleaned {data_type} data to save")
        return

    project_root = get_project_root()

    # Save CSV to logs directory
    logs_dir = os.path.join(project_root, 'data', 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    csv_filename = f'{data_type}_data_cleaned.csv'
    csv_path = os.path.join(logs_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved cleaned {data_type} data to CSV: {csv_path}")

    # Save JSON to processed directory
    processed_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    json_filename = f'{data_type}_data_cleaned.json'
    json_path = os.path.join(processed_dir, json_filename)
    try:
        records = df.to_dict(orient='records')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4, default=str)
        logging.info(f"Saved cleaned {data_type} data to JSON: {json_path}")
    except Exception as e:
        logging.error(f"Error saving {data_type} data to JSON: {e}")


def load_json_data(file_path):
    """Load JSON data from the given file path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None


def clean_stock_data(stock_data):
    """ Clean stock data from all_stock_data.json. """
    if not stock_data:
        return pd.DataFrame()

    # Convert to DataFrame if it's not already
    if isinstance(stock_data, dict):
        flattened_data = []
        for ticker, data in stock_data.items():
            for record in data:
                record['ticker'] = ticker
                flattened_data.append(record)
        df = pd.DataFrame(flattened_data)
    elif isinstance(stock_data, list):
        df = pd.DataFrame(stock_data)
    else:
        df = stock_data.copy()

    # Ensure 'Date' column is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Remove rows with missing essential data
    essential_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    essential_columns = [col for col in essential_columns if col in df.columns]
    df = df.dropna(subset=essential_columns)

    # Handle outliers in price columns
    price_columns = ['Open', 'High', 'Low', 'Close']
    for col in price_columns:
        if col in df.columns:
            # Replace extreme outliers with NaN and then forward fill
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df.loc[(df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR)), col] = np.nan
            df[col] = df[col].ffill()

    # Calculate daily returns
    if 'Daily_Return' not in df.columns and 'Close' in df.columns:
        df['Daily_Return'] = df.groupby('ticker')['Close'].pct_change()

    # Add day of week and month features
    if 'Date' in df.columns:
        df['Day_of_Week'] = df['Date'].dt.day_name()
        df['Month'] = df['Date'].dt.month_name()

    return df


import pandas as pd
import logging
from datetime import datetime
import numpy as np


def clean_news_data(news_data):
    """ Clean news data from news_data.json. """
    # Log the initial data state
    if not news_data:
        logging.warning("No news data to clean (received empty dataset)")
        # Return a dummy record to prevent downstream processing errors
        dummy_df = pd.DataFrame([{
            'title': 'Data Cleaning Error',
            'description': 'No news data was available to clean',
            'publishedAt': datetime.now().isoformat(),
            'source': 'System',
            'sentiment_score': 0.0,
            'processed_date': datetime.now().strftime('%Y-%m-%d')
        }])
        return dummy_df

    # Log the type and size of incoming data
    logging.info(
        f"News data type: {type(news_data)}, size: {len(news_data) if isinstance(news_data, list) else 'unknown'}")

    # Check if news_data is already a DataFrame
    if isinstance(news_data, pd.DataFrame):
        df = news_data
        logging.info(f"News data was provided as DataFrame with shape: {df.shape}")
    else:
        # Try to convert to DataFrame
        try:
            df = pd.DataFrame(news_data)
            logging.info(f"Successfully converted news data to DataFrame with shape: {df.shape}")
        except Exception as e:
            logging.error(f"Failed to convert news data to DataFrame: {e}")
            # Return a dummy DataFrame to prevent downstream errors
            dummy_df = pd.DataFrame([{
                'title': 'Data Conversion Error',
                'description': f'Failed to convert news data to DataFrame: {str(e)}',
                'publishedAt': datetime.now().isoformat(),
                'source': 'System',
                'sentiment_score': 0.0,
                'processed_date': datetime.now().strftime('%Y-%m-%d')
            }])
            return dummy_df

    # Check if DataFrame is empty
    if df.empty:
        logging.warning("News DataFrame is empty after conversion")
        # Return a dummy record
        dummy_df = pd.DataFrame([{
            'title': 'Empty DataFrame',
            'description': 'News data converted to an empty DataFrame',
            'publishedAt': datetime.now().isoformat(),
            'source': 'System',
            'sentiment_score': 0.0,
            'processed_date': datetime.now().strftime('%Y-%m-%d')
        }])
        return dummy_df

    # Log the columns we have to work with
    logging.info(f"News DataFrame columns: {df.columns.tolist()}")

    # Ensure required columns exist (add placeholders if missing)
    required_columns = ['title', 'description', 'publishedAt', 'source', 'url']
    for col in required_columns:
        if col not in df.columns:
            logging.warning(f"Missing required column '{col}', adding placeholder")
            df[col] = f"No {col} available"

    # Convert publishedAt to datetime
    if 'publishedAt' in df.columns:
        try:
            # First, check some sample values
            sample_dates = df['publishedAt'].head().tolist()
            logging.info(f"Sample publishedAt values: {sample_dates}")

            # Try to parse dates without specifying format
            df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')

            # Log how many dates were successfully parsed
            valid_dates = df['publishedAt'].notna().sum()
            invalid_dates = df['publishedAt'].isna().sum()
            logging.info(
                f"Successfully parsed {valid_dates} out of {len(df)} publishedAt dates. Invalid: {invalid_dates}")

            if invalid_dates > 0:
                # Try common news API formats for invalid dates
                formats_to_try = [
                    '%Y-%m-%dT%H:%M:%SZ',  # ISO format: "2023-05-11T15:30:00Z"
                    '%Y-%m-%dT%H:%M:%S.%fZ',  # ISO with milliseconds: "2023-05-11T15:30:00.000Z"
                    '%Y-%m-%d %H:%M:%S',  # Simple format: "2023-05-11 15:30:00"
                    '%a, %d %b %Y %H:%M:%S %Z',  # RFC 822 format: "Thu, 11 May 2023 15:30:00 GMT"
                ]

                # Try each format and log results
                for date_format in formats_to_try:
                    try:
                        # Get indices of invalid dates
                        invalid_indices = df[df['publishedAt'].isna()].index
                        if len(invalid_indices) > 0:
                            # Try to parse with specific format
                            temp_dates = pd.to_datetime(
                                df.loc[invalid_indices, 'publishedAt'],
                                format=date_format,
                                errors='coerce'
                            )
                            valid_count = temp_dates.notna().sum()

                            if valid_count > 0:
                                logging.info(f"Format '{date_format}' successfully parsed {valid_count} invalid dates")
                                # Apply the format to invalid dates
                                df.loc[invalid_indices, 'publishedAt'] = temp_dates
                    except Exception as e:
                        logging.warning(f"Error trying format '{date_format}': {e}")

                # For any remaining NaT values, use current date
                remaining_invalid = df['publishedAt'].isna().sum()
                if remaining_invalid > 0:
                    logging.warning(f"Setting {remaining_invalid} unparseable dates to current date")
                    df.loc[df['publishedAt'].isna(), 'publishedAt'] = datetime.now()

        except Exception as e:
            logging.error(f"Error processing publishedAt dates: {e}")
            # Set all dates to current date if there was an error
            df['publishedAt'] = datetime.now()
    else:
        logging.warning("No 'publishedAt' column found in news data, creating one")
        df['publishedAt'] = datetime.now()

    # Remove duplicates based on title and description
    original_count = len(df)
    try:
        if 'title' in df.columns and 'description' in df.columns:
            df = df.drop_duplicates(subset=['title', 'description'])
            logging.info(f"Removed {original_count - len(df)} duplicate articles based on title and description")
        elif 'title' in df.columns:
            df = df.drop_duplicates(subset='title')
            logging.info(f"Removed {original_count - len(df)} duplicate articles based on title")
        elif 'description' in df.columns:
            df = df.drop_duplicates(subset='description')
            logging.info(f"Removed {original_count - len(df)} duplicate articles based on description")
    except Exception as e:
        logging.error(f"Error removing duplicates: {e}")

    # Remove articles with empty title and description
    original_count = len(df)
    try:
        if 'title' in df.columns and 'description' in df.columns:
            # First, fill NaN values with empty strings to avoid errors with str methods
            df['title'] = df['title'].fillna('')
            df['description'] = df['description'].fillna('')

            # Now filter out rows where both title and description are empty
            df = df[~((df['title'].str.strip() == '') & (df['description'].str.strip() == ''))]
            logging.info(f"Removed {original_count - len(df)} articles with empty title and description")
        elif 'title' in df.columns:
            df['title'] = df['title'].fillna('')
            df = df[~(df['title'].str.strip() == '')]
            logging.info(f"Removed {original_count - len(df)} articles with empty title")
    except Exception as e:
        logging.error(f"Error removing empty articles: {e}")

    # Add processed date field
    df['processed_date'] = datetime.now().strftime('%Y-%m-%d')

    # Add placeholder for sentiment analysis
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = 0.0

    # Ensure all remaining NaN values are handled
    df = df.fillna({
        'author': 'Unknown',
        'source': 'Unknown',
        'url': '',
        'country': 'N/A',
    })

    # Log final DataFrame shape
    logging.info(f"News DataFrame shape after cleaning: {df.shape}")

    # If DataFrame is still empty after all processing, create a dummy record
    if df.empty:
        logging.warning("DataFrame is empty after cleaning, creating a dummy record")
        df = pd.DataFrame([{
            'title': 'No Valid News Data',
            'description': 'All news data was filtered out during cleaning',
            'publishedAt': datetime.now(),
            'source': 'System',
            'url': '',
            'author': 'System',
            'country': 'N/A',
            'sentiment_score': 0.0,
            'processed_date': datetime.now().strftime('%Y-%m-%d')
        }])

    return df


def clean_economic_data(economic_data):
    """ Clean economic data from economic_data.json. """
    if not economic_data:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(economic_data)

    # Handle missing values
    if 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # Convert date strings to datetime objects
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Create year column
    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = df['date'].dt.year

    # Create country_code column if it doesn't exist
    if 'country_code' not in df.columns and 'country' in df.columns:
        df['country_code'] = df['country']

    # Add human-readable country names
    country_codes = {
        'US': 'United States',
        'GB': 'United Kingdom',
        'CN': 'China',
        'IN': 'India'
    }
    if 'country_code' in df.columns:
        df['country_name'] = df['country_code'].map(country_codes)

    # Sort by date (newest first) and country
    sort_columns = [col for col in ['date', 'country'] if col in df.columns]
    if sort_columns:
        df = df.sort_values(by=sort_columns, ascending=[False, True])

    return df


def clean_twitter_data(twitter_data):
    """ Clean Twitter data from twitter_data.json. """

    if not twitter_data:
        logging.warning("No Twitter data to clean")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(twitter_data)

    # Log the shape and columns for debugging
    logging.info(f"Twitter DataFrame shape before cleaning: {df.shape}")
    logging.info(f"Twitter DataFrame columns: {df.columns.tolist()}")

    # Handle date parsing for 'created_at'
    if 'created_at' in df.columns and not df.empty:
        # Sample logging to debug date format
        sample_dates = df['created_at'].head().tolist()
        logging.info(f"Sample created_at values: {sample_dates}")

        # Define Twitter date formats to try (in order of likelihood)
        formats_to_try = [
            '%a %b %d %H:%M:%S +0000 %Y',  # "Sun May 11 12:58:38 +0000 2025"
            '%a %b %d %H:%M:%S %z %Y',  # With timezone in different format
            '%Y-%m-%dT%H:%M:%S.%fZ',  # ISO format with milliseconds
            '%Y-%m-%dT%H:%M:%SZ',  # ISO format without milliseconds
            '%Y-%m-%d %H:%M:%S',  # Simple format
        ]

        # Create a new column to store parsed dates
        df['parsed_date'] = pd.NaT

        # Try each format until we've parsed all dates
        for date_format in formats_to_try:
            # Only attempt to parse dates that haven't been parsed yet
            mask = df['parsed_date'].isna()
            if not any(mask):
                break  # All dates parsed, exit the loop

            try:
                # Try to parse unparsed dates with current format
                df.loc[mask, 'parsed_date'] = pd.to_datetime(
                    df.loc[mask, 'created_at'],
                    format=date_format,
                    errors='coerce'
                )
                success_count = (~df['parsed_date'].isna() & mask).sum()
                logging.info(f"Format '{date_format}' successfully parsed {success_count} dates")
            except Exception as e:
                logging.warning(f"Error trying format '{date_format}': {e}")

        # Report final parsing results
        parsed_count = (~df['parsed_date'].isna()).sum()
        total_count = len(df)
        logging.info(f"Successfully parsed {parsed_count} out of {total_count} created_at dates")

        if parsed_count < total_count:
            logging.warning(f"Failed to parse {total_count - parsed_count} dates")
            # Log some examples of dates that couldn't be parsed
            if any(df['parsed_date'].isna()):
                unparsed_examples = df.loc[df['parsed_date'].isna(), 'created_at'].head(5).tolist()
                logging.warning(f"Examples of unparsed dates: {unparsed_examples}")

        # Replace the original created_at with the parsed dates
        df['created_at'] = df['parsed_date']
        df.drop('parsed_date', axis=1, inplace=True)

        # Drop rows with unparsed dates
        invalid_dates = df['created_at'].isna()
        if any(invalid_dates):
            df = df[~invalid_dates]
            logging.info(f"Removed {invalid_dates.sum()} tweets with invalid dates")

    else:
        logging.warning("No 'created_at' column found in Twitter data")

    # Remove tweets with empty text
    if 'text' in df.columns:
        empty_texts = (df['text'].isna()) | (df['text'].str.strip() == '')
        df = df[~empty_texts]
        logging.info(f"Removed {empty_texts.sum()} tweets with empty text")

    # Remove duplicate tweets based on text
    if 'text' in df.columns:
        duplicates = df.duplicated(subset=['text'])
        df = df[~duplicates]
        logging.info(f"Removed {duplicates.sum()} duplicate tweets")

    # Add language flag for non-English tweets
    if 'lang' in df.columns:
        df['is_english'] = df['lang'] == 'en'
        logging.info(f"English tweets: {df['is_english'].sum()}, Non-English: {(~df['is_english']).sum()}")

    # Clean up text field
    if 'text' in df.columns:
        df['cleaned_text'] = df['text'].str.replace(r'http\S+|www\S+', '', regex=True)
        df['cleaned_text'] = df['cleaned_text'].str.replace(r'@\w+', '', regex=True)
        df['cleaned_text'] = df['cleaned_text'].str.replace(r'#\w+', '', regex=True)
        df['cleaned_text'] = df['cleaned_text'].str.replace('\n', ' ', regex=False)
        df['cleaned_text'] = df['cleaned_text'].str.strip().str.replace(r'\s+', ' ', regex=True)

    # Add processed date field
    df['processed_date'] = datetime.now().strftime('%Y-%m-%d')

    # Log final DataFrame shape
    logging.info(f"Twitter DataFrame shape after cleaning: {df.shape}")

    return df

def clean_reddit_data(reddit_data):
    """ Clean Reddit data from reddit_data.json. """
    if not reddit_data:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(reddit_data)

    # Convert timestamps to datetime
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], unit='s', errors='coerce')

    # Remove posts with empty title or missing essential fields
    essential_columns = ['title', 'subreddit']
    essential_columns = [col for col in essential_columns if col in df.columns]
    if essential_columns:
        df = df.dropna(subset=essential_columns)

    # Process comments if available
    if 'comments' in df.columns:
        df['valid_comments_count'] = df['comments'].apply(
            lambda comments: sum(1 for c in comments if isinstance(c, dict) and c.get('comment_text'))
        )

        def combine_post_comments(row):
            if 'title' not in row:
                return ""
            text = row.get('title', '')
            if 'comments' in row and isinstance(row['comments'], list):
                for comment in row['comments']:
                    if isinstance(comment, dict) and 'comment_text' in comment:
                        text += " " + comment['comment_text']
            return text

        df['combined_text'] = df.apply(combine_post_comments, axis=1)

    # Add processed date field
    df['processed_date'] = datetime.now().strftime('%Y-%m-%d')

    return df


def apply_sentiment_analysis(df, sentiment_model='vader'):
    """ Apply sentiment analysis to text data, including comments. """
    if sentiment_model == 'vader':
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()

            # Process post text
            text_columns = []
            if 'text' in df.columns:
                text_columns.append('text')
            elif 'cleaned_text' in df.columns:
                text_columns.append('cleaned_text')
            elif 'title' in df.columns:
                text_columns.append('title')
            elif 'description' in df.columns:
                text_columns.append('description')
            elif 'combined_text' in df.columns:
                text_columns.append('combined_text')

            for col in text_columns:
                if col in df.columns:
                    df[f'{col}_neg'] = df[col].apply(
                        lambda text: analyzer.polarity_scores(str(text))['neg'] if pd.notna(text) else 0
                    )
                    df[f'{col}_neu'] = df[col].apply(
                        lambda text: analyzer.polarity_scores(str(text))['neu'] if pd.notna(text) else 0
                    )
                    df[f'{col}_pos'] = df[col].apply(
                        lambda text: analyzer.polarity_scores(str(text))['pos'] if pd.notna(text) else 0
                    )
                    df[f'{col}_compound'] = df[col].apply(
                        lambda text: analyzer.polarity_scores(str(text))['compound'] if pd.notna(text) else 0
                    )

            # Process comments
            if 'comments' in df.columns:
                for idx, row in df.iterrows():
                    if isinstance(row['comments'], list):
                        for comment in row['comments']:
                            if 'comment_text' in comment:
                                text = comment['comment_text']
                                scores = analyzer.polarity_scores(str(text))
                                comment['comment_neg'] = scores['neg']
                                comment['comment_neu'] = scores['neu']
                                comment['comment_pos'] = scores['pos']
                                comment['comment_compound'] = scores['compound']

            # Add consolidated sentiment for posts
            if 'text_compound' in df.columns:
                df['sentiment'] = pd.cut(
                    df['text_compound'], bins=[-1, -0.5, 0.1, 0.5, 1],
                    labels=['negative', 'slightly negative', 'neutral', 'positive']
                )
            elif 'title_compound' in df.columns:
                df['sentiment'] = pd.cut(
                    df['title_compound'], bins=[-1, -0.5, 0.1, 0.5, 1],
                    labels=['negative', 'slightly negative', 'neutral', 'positive']
                )
        except ImportError:
            logging.error("VADER Sentiment Analysis package not installed. Skipping sentiment analysis.")
    elif sentiment_model == 'textblob':
        try:
            from textblob import TextBlob

            # Process post text
            text_columns = []
            if 'text' in df.columns:
                text_columns.append('text')
            elif 'cleaned_text' in df.columns:
                text_columns.append('cleaned_text')
            elif 'title' in df.columns:
                text_columns.append('title')
            elif 'description' in df.columns:
                text_columns.append('description')
            elif 'combined_text' in df.columns:
                text_columns.append('combined_text')

            for col in text_columns:
                if col in df.columns:
                    df[f'{col}_polarity'] = df[col].apply(
                        lambda text: TextBlob(str(text)).sentiment.polarity if pd.notna(text) else 0
                    )
                    df[f'{col}_subjectivity'] = df[col].apply(
                        lambda text: TextBlob(str(text)).sentiment.subjectivity if pd.notna(text) else 0
                    )

            # Process comments
            if 'comments' in df.columns:
                for idx, row in df.iterrows():
                    if isinstance(row['comments'], list):
                        for comment in row['comments']:
                            if 'comment_text' in comment:
                                text = comment['comment_text']
                                blob = TextBlob(str(text))
                                comment['comment_polarity'] = blob.sentiment.polarity
                                comment['comment_subjectivity'] = blob.sentiment.subjectivity

            # Add consolidated sentiment for posts
            if 'text_polarity' in df.columns:
                df['sentiment'] = pd.cut(
                    df['text_polarity'], bins=[-1, -0.2, 0.2, 1],
                    labels=['negative', 'neutral', 'positive']
                )
            elif 'title_polarity' in df.columns:
                df['sentiment'] = pd.cut(
                    df['title_polarity'], bins=[-1, -0.2, 0.2, 1],
                    labels=['negative', 'neutral', 'positive']
                )
        except ImportError:
            logging.error("TextBlob package not installed. Skipping sentiment analysis.")
    else:
        logging.error(f"Unsupported sentiment model: {sentiment_model}. Skipping sentiment analysis.")

    return df


def process_raw_data(data_type):
    """
    Process raw data of a specific type, clean it, and save it.
    Returns the cleaned data in a MongoDB-compatible format (list of dictionaries).
    """
    import traceback

    logging.info(f"Processing raw {data_type} data")

    def get_project_root():
        # Get the project root directory
        # Adjust this based on your project structure
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def load_json_data(file_path):
        """Load and return JSON data from the given file path."""
        try:
            if not os.path.isfile(file_path):
                logging.error(f"File not found: {file_path}")
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Enhanced validation
            if data is None:
                logging.error(f"Loaded data is None from {file_path}")
                return None

            if isinstance(data, list):
                if not data:
                    logging.warning(f"Loaded data is an empty list from {file_path}")
                    return []
                logging.info(f"Successfully loaded {len(data)} records from {file_path}")
            else:
                logging.info(f"Successfully loaded JSON data from {file_path} (non-list type: {type(data)})")

            return data
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error when loading {file_path}: {e}")
            # Try to debug the file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # Read first 1000 chars for debugging
                logging.error(f"File content start: {content}")
            except Exception as file_e:
                logging.error(f"Could not read file for debugging: {file_e}")
            return None
        except Exception as e:
            logging.error(f"Error loading JSON from {file_path}: {e}")
            return None

    def save_cleaned_data(cleaned_data, data_type):
        """Save cleaned data to a JSON file."""
        try:
            project_root = get_project_root()
            cleaned_data_path = os.path.join(project_root, 'data', 'cleaned', f'{data_type}_data_cleaned.json')

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)

            # If data is a DataFrame, convert to list of dictionaries
            if isinstance(cleaned_data, pd.DataFrame):
                data_to_save = cleaned_data.to_dict('records')
            else:
                data_to_save = cleaned_data

            with open(cleaned_data_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=4, default=str)

            logging.info(f"Saved cleaned {data_type} data to {cleaned_data_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving cleaned {data_type} data: {e}")
            return False

    # Import cleaning functions or define them here
    project_root = get_project_root()

    # Handle special case for stock data filename
    if data_type == 'stock':
        raw_data_path = os.path.join(project_root, 'data', 'raw', 'all_stock_data.json')
    else:
        raw_data_path = os.path.join(project_root, 'data', 'raw', f'{data_type}_data.json')

    if not os.path.exists(raw_data_path):
        logging.error(f"Raw data file not found: {raw_data_path}")
        return []  # Return empty list if file not found

    # Load data with enhanced error handling and validation
    raw_data = load_json_data(raw_data_path)
    if raw_data is None:
        logging.error(f"Failed to load data from {raw_data_path}")
        return []  # Return empty list if data couldn't be loaded

    # Special handling for Twitter data to ensure it's in the right format
    if data_type == 'twitter':
        if not isinstance(raw_data, list):
            logging.error(f"Twitter data is not a list: {type(raw_data)}")
            # Try to convert to list if it's not already
            if hasattr(raw_data, 'items'):  # Dictionary-like
                raw_data = list(raw_data.values())
                logging.info("Converted dictionary-like data to list")
            else:
                # Try to wrap in a list if it's a single item
                try:
                    raw_data = [raw_data]
                    logging.info("Wrapped single item in a list")
                except:
                    logging.error("Could not convert Twitter data to a list")
                    return []

        if not raw_data:  # Empty list
            logging.error("Twitter data is an empty list")
            return []

        # Log some sample data for debugging
        if len(raw_data) > 0:
            sample = raw_data[0]
            logging.info(f"Sample Twitter data: {sample}")
            logging.info(f"'created_at' format: {sample.get('created_at', 'Not present')}")

    # Log the raw data type and size
    logging.info(
        f"Raw {data_type} data type: {type(raw_data)}, size: {len(raw_data) if isinstance(raw_data, list) else 'unknown'}")

    # Determine the appropriate cleaning function
    cleaners = {
        'stock': clean_stock_data,
        'news': clean_news_data,
        'economic': clean_economic_data,
        'twitter': clean_twitter_data,
        'reddit': clean_reddit_data
    }

    cleaner_func = cleaners.get(data_type)
    if not cleaner_func:
        logging.error(f"No cleaner function available for {data_type}")
        return []  # Return empty list if no cleaner available

    # Clean data with additional error handling
    try:
        cleaned_data = cleaner_func(raw_data)
        logging.info(f"Cleaning function for {data_type} executed successfully")
    except Exception as e:
        logging.error(f"Error during cleaning {data_type} data: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return []

    # Check if cleaned data is empty
    if isinstance(cleaned_data, pd.DataFrame):
        if cleaned_data.empty:
            logging.warning(f"Cleaned {data_type} DataFrame is empty, nothing to process further")
            return []
        else:
            logging.info(f"Cleaned {data_type} DataFrame shape: {cleaned_data.shape}")
    elif not cleaned_data:
        logging.warning(f"Cleaned {data_type} data is empty (non-DataFrame)")
        return []

    # Apply sentiment analysis for text-based data types
    if data_type in ['news', 'twitter', 'reddit']:
        try:
            cleaned_data = apply_sentiment_analysis(cleaned_data, sentiment_model='vader')
            logging.info(f"Sentiment analysis applied to {data_type} data")
        except Exception as e:
            logging.error(f"Error applying sentiment analysis to {data_type} data: {e}")
            logging.error(traceback.format_exc())
            # Continue without sentiment analysis

    # Save cleaned data
    save_cleaned_data(cleaned_data, data_type)

    logging.info(f"Finished processing {data_type} data")

    # Convert DataFrame to list of dictionaries for MongoDB
    mongo_data = []
    if isinstance(cleaned_data, pd.DataFrame) and not cleaned_data.empty:
        # Convert DataFrame to dict of records
        try:
            for _, row in cleaned_data.iterrows():
                record = {}
                for key, value in row.items():
                    # Handle different data types appropriately
                    if isinstance(value, pd.Timestamp) or isinstance(value, datetime):
                        record[key] = value.isoformat()
                    elif isinstance(value, np.ndarray):
                        # Handle numpy arrays - convert to list
                        record[key] = value.tolist() if value.size > 0 else None
                    elif pd.api.types.is_scalar(value) and pd.isna(value):
                        # Only check isna() on scalar values
                        record[key] = None
                    else:
                        # Keep other values as is
                        record[key] = value
                mongo_data.append(record)

            logging.info(f"Successfully converted {data_type} DataFrame to {len(mongo_data)} MongoDB documents")
        except Exception as e:
            logging.error(f"Error converting {data_type} DataFrame to MongoDB format: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return []
    else:
        logging.warning(f"Cleaned {data_type} data is not a valid DataFrame or is empty")
        return []

    return mongo_data

#
# if __name__ == "__main__":
#     # Process all data types
#     data_types = ['stock', 'news', 'economic', 'twitter', 'reddit']
#     for data_type in data_types:
#         try:
#             process_raw_data(data_type)
#         except Exception as e:
#             logging.error(f"Error processing {data_type} data: {e}")
#     logging.info("All data processing complete")
