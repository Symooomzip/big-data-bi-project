import requests
import json
import os
import logging
from bs4 import BeautifulSoup
from datetime import datetime


def get_reuters_data(search_term):
    """Scrape news data from Reuters based on a search term."""
    try:
        url = f"https://www.reuters.com/search/news?blob={search_term}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            logging.error(f"Reuters API returned status code {response.status_code}")
            return []

        soup = BeautifulSoup(response.content, "html.parser")

        articles = []
        for article in soup.find_all('div', class_='search-result'):
            try:
                title_elem = article.find('h3')
                title = title_elem.get_text() if title_elem else "No title available"

                link_elem = article.find('a')
                link = link_elem['href'] if link_elem and 'href' in link_elem.attrs else ""

                desc_elem = article.find('p')
                description = desc_elem.get_text() if desc_elem else "No description available"

                articles.append({
                    'title': title,
                    'description': description,
                    'url': f"https://www.reuters.com{link}" if link else "",
                    'source': 'Reuters',
                    'publishedAt': datetime.now().isoformat()  # Use current time since Reuters doesn't provide date
                })
            except Exception as e:
                logging.error(f"Error parsing Reuters article: {e}")
                continue

        logging.info(f"Successfully scraped {len(articles)} articles from Reuters")
        return articles
    except Exception as e:
        logging.error(f"Error scraping Reuters: {e}")
        return []


def collect_news_data():
    """Collect news data from News API and Reuters."""
    try:
        # Get the absolute path to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        logging.info(f"Project root determined as: {project_root}")

        # Build the path to the config folder
        config_path = os.path.join(project_root, 'config', 'news_config.json')
        logging.info(f"Looking for config at: {config_path}")

        # Check if the file exists
        if not os.path.exists(config_path):
            logging.error(f"Config file not found: {config_path}")
            # Use a fallback approach - check environment variable
            news_api_key = os.environ.get('NEWS_API_KEY')
            if not news_api_key:
                logging.error("No NEWS_API_KEY found in environment variables either")
                # Try to continue with Reuters scraping only
                news_api_key = None
            else:
                logging.info("Using NEWS_API_KEY from environment variables")
        else:
            # Load News API credentials from the JSON config file
            try:
                with open(config_path, 'r') as f:
                    news_config = json.load(f)

                # Extract API key with proper error handling
                news_api_key = news_config.get('NEWS_API_KEY')
                if not news_api_key:
                    # Check for alternative key names
                    news_api_key = news_config.get('API_KEY') or news_config.get('NEWSAPI_KEY')
                    if not news_api_key:
                        logging.error("No valid API key found in config file")
                    else:
                        logging.info("Found API key under alternative name in config")
                else:
                    logging.info("Successfully loaded NEWS_API_KEY from config")
            except Exception as e:
                logging.error(f"Error loading config file: {e}")
                news_api_key = os.environ.get('NEWS_API_KEY')
                if news_api_key:
                    logging.info("Using NEWS_API_KEY from environment variables as fallback")
                else:
                    logging.error("No fallback NEWS_API_KEY found in environment variables")

        # Initialize all_news_articles list
        all_news_articles = []

        # Define event types (can be expanded dynamically)
        event_types = ['global economic crisis', 'natural disasters', 'political events', 'technology news']

        # News API collection - only if we have an API key
        if news_api_key:
            # News API endpoint
            url = 'https://newsapi.org/v2/everything'

            for event in event_types:
                try:
                    # News API parameters
                    params = {
                        'q': event,  # Search query
                        'apiKey': news_api_key,
                        'language': 'en',
                        'pageSize': 10,  # Limit results for testing
                    }

                    # Fetch data from News API
                    logging.info(f"Fetching news for topic: {event}")
                    response = requests.get(url, params=params, timeout=10)

                    if response.status_code != 200:
                        logging.error(f"News API returned status code {response.status_code} for query '{event}'")
                        logging.error(f"Response content: {response.text[:200]}...")  # Log beginning of response
                        continue

                    news_data = response.json()

                    # Check if we got expected response structure
                    if 'articles' not in news_data:
                        logging.error(f"Unexpected response structure from News API: {news_data.keys()}")
                        if 'message' in news_data:
                            logging.error(f"API error message: {news_data['message']}")
                        continue

                    # Log how many articles were received
                    articles_count = len(news_data.get('articles', []))
                    logging.info(f"Received {articles_count} articles from News API for '{event}'")

                    # Process News API response
                    for article in news_data.get('articles', []):
                        if not article:
                            continue

                        news_article = {
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'publishedAt': article.get('publishedAt', ''),
                            'author': article.get('author', ''),
                            'source': article.get('source', {}).get('name', 'Unknown'),
                            'url': article.get('url', ''),
                            'country': article.get('country', 'N/A')
                        }
                        all_news_articles.append(news_article)
                except Exception as e:
                    logging.error(f"Error processing News API data for event '{event}': {e}")
        else:
            logging.warning("Skipping News API collection due to missing API key")

        # Collect news from Reuters for each event type
        for event in event_types:
            try:
                logging.info(f"Scraping Reuters for topic: {event}")
                reuters_data = get_reuters_data(event)
                all_news_articles.extend(reuters_data)
            except Exception as e:
                logging.error(f"Error collecting Reuters data for event '{event}': {e}")

        # Check if we collected any articles
        if not all_news_articles:
            logging.warning("No news articles collected from any source")
            # Create a minimal test article to avoid empty data
            all_news_articles = [{
                'title': 'Test Article - Data Collection Issue',
                'description': 'This is a placeholder article created because no real news could be collected.',
                'publishedAt': datetime.now().isoformat(),
                'author': 'System',
                'source': 'Internal',
                'url': '',
                'country': 'N/A'
            }]
            logging.info("Added placeholder article to prevent empty dataset")

        # Save collected news data to a JSON file
        output_path = os.path.join(project_root, 'data', 'raw', 'news_data.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the directory exists

        # Save the detailed news data to the output file
        with open(output_path, 'w') as f:
            json.dump(all_news_articles, f, indent=4)

        logging.info(f"Collected and saved {len(all_news_articles)} news articles to {output_path}")

        # Return the collected data
        return all_news_articles
    except Exception as e:
        logging.error(f"Critical error in news data collection: {e}")
        import traceback
        logging.error(traceback.format_exc())

        # Return a minimal dataset to prevent downstream errors
        return [{
            'title': 'Error in News Collection',
            'description': f'An error occurred during news collection: {str(e)}',
            'publishedAt': datetime.now().isoformat(),
            'author': 'System',
            'source': 'Error',
            'url': '',
            'country': 'N/A'
        }]

#
# # For standalone testing
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     news_data = collect_news_data()
#     print(f"Collected {len(news_data)} news articles")