import praw
import json
import os
import time


def collect_reddit_data():
    # Get the project root path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Path to the Reddit API config file
    config_path = os.path.join(project_root, 'config', 'reddit_config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Read Reddit credentials from the config file
    with open(config_path, 'r') as f:
        reddit_config = json.load(f)

    # Initialize Reddit instance with praw
    reddit = praw.Reddit(
        client_id=reddit_config['REDDIT_CLIENT_ID'],
        client_secret=reddit_config['REDDIT_CLIENT_SECRET'],
        user_agent=reddit_config['REDDIT_USER_AGENT']
    )

    # Define event keywords to search for
    event_keywords = ['global economic crisis', 'natural disaster', 'political unrest', 'inflation trends']
    all_reddit_data = []

    # Iterate over each keyword
    for term in event_keywords:
        print(f"Searching Reddit for: {term}")
        subreddit = reddit.subreddit('all')
        posts = subreddit.search(term, limit=50)  # Limit to 50 posts per term

        # Iterate over the posts
        for post in posts:
            if post.score > 10 and not post.over_18:  # Filter low-score and NSFW posts
                post_data = {
                    'title': post.title,
                    'url': post.url,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'subreddit': post.subreddit.display_name,
                    'author': str(post.author),
                    'created_at': post.created_utc,
                    'event_type': term,
                    'comments': [],
                    'sentiment': None  # Placeholder for sentiment analysis
                }

                # Attempt to retrieve comments
                try:
                    post.comments.replace_more(limit=0)  # Avoid replacing more comments
                    top_comments = sorted(post.comments.list(), key=lambda x: x.score, reverse=True)[
                                   :5]  # Top 5 comments

                    for comment in top_comments:
                        comment_data = {
                            'comment_text': comment.body,
                            'comment_score': comment.score,
                            'comment_author': str(comment.author),
                            'comment_created_at': comment.created_utc,
                            'sentiment': None  # Placeholder for sentiment analysis
                        }
                        post_data['comments'].append(comment_data)

                except Exception as e:
                    print(f"Error processing comments for post {post.id}: {e}")

                # Add the post data to the list
                all_reddit_data.append(post_data)

    # Output path for saving the collected data
    output_path = os.path.join(project_root, 'data', 'raw', 'reddit_data.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write the collected data to a JSON file
    with open(output_path, 'w') as f:
        json.dump(all_reddit_data, f, indent=4)

    print(f"Collected {len(all_reddit_data)} Reddit posts with comments.")
