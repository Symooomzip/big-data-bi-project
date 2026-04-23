#!/usr/bin/env python
"""
Main entry point for the stock market data collection and prediction system.
This script coordinates data collection, processing, model training, and prediction.
"""
import os
import sys
import argparse
import logging
import threading
import time
import schedule
from datetime import datetime

# Set up proper path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import components
from src.automation.schedule_collection import collect_and_process_all_data
from src.machine_learning.train import train_models
from src.machine_learning.predict import predict_stock, interactive_predict_stock
import src.config as config

# Set up logging
log_file = os.path.join(project_root, 'logs', 'main_system.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add console handler to see logs in terminal
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


def run_scheduled_jobs():
    """
    Run the scheduled jobs in a separate thread
    """
    logging.info("Starting scheduled job runner...")

    # Schedule data collection every 15 minutes
    schedule.every(15).minutes.do(collect_and_process_all_data)

    # Schedule model training every hour (adjust frequency as needed)
    schedule.every(1).hours.do(train_models_for_all_tickers)

    # Keep running the schedule
    while True:
        schedule.run_pending()
        time.sleep(1)


def train_models_for_all_tickers():
    """
    Train models for all tracked tickers
    """
    logging.info("Starting scheduled model training...")

    # Get list of tickers from config or use defaults
    tickers = getattr(config, 'TRACKED_TICKERS', ["AAPL", "MSFT", "GOOGL", "SPY"])

    for ticker in tickers:
        try:
            logging.info(f"Training models for {ticker}...")
            train_models(
                ticker=ticker,
                days_back=getattr(config, 'TRAINING_DAYS_BACK', 365),
                save_models=True
            )
            logging.info(f"Model training completed for {ticker}")
        except Exception as e:
            logging.error(f"Error training models for {ticker}: {e}")

    logging.info("All scheduled model training completed")


def run_predictions(tickers=None, days_ahead=5):
    """
    Run predictions for specified tickers

    Args:
        tickers (list): List of tickers to predict (None for all tracked tickers)
        days_ahead (int): Number of days ahead to predict
    """
    if tickers is None:
        tickers = getattr(config, 'TRACKED_TICKERS', ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"])

    results = {}

    for ticker in tickers:
        try:
            logging.info(f"Making predictions for {ticker}...")
            prediction = predict_stock(
                ticker=ticker,
                days_back=30,
                model_name=None,  # Use best model
                prediction_days=days_ahead,
                model_dir=os.path.join(project_root, "src", "machine_learning", "models")
            )

            if prediction:
                results[ticker] = prediction
                logging.info(f"Prediction completed for {ticker}")

                # Save to MongoDB
                from src.mongodb.mongo_handler import MongoDBHandler
                mongo_handler = MongoDBHandler()
                mongo_handler.upsert_data([prediction], 'predictions')
                logging.info(f"Saved {ticker} predictions to MongoDB")
            else:
                logging.warning(f"No prediction results for {ticker}")
        except Exception as e:
            logging.error(f"Error making predictions for {ticker}: {e}")

    return results


def main():
    """
    Main function to parse arguments and run the appropriate function
    """
    parser = argparse.ArgumentParser(
        description='Stock Market Data Collection and Prediction System'
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Subparser for starting the scheduled service
    service_parser = subparsers.add_parser('service', help='Run as a scheduled service')

    # Subparser for collecting data once
    collect_parser = subparsers.add_parser('collect', help='Run data collection once')

    # Subparser for training models
    train_parser = subparsers.add_parser('train', help='Train prediction models')
    train_parser.add_argument('--ticker', type=str, help='Stock ticker to train for')
    train_parser.add_argument('--days-back', type=int, default=365,
                              help='Days of historical data to use for training')

    # Subparser for making predictions
    predict_parser = subparsers.add_parser('predict', help='Make stock predictions')
    predict_parser.add_argument('--ticker', type=str, nargs='+', help='Stock ticker(s) to predict')
    predict_parser.add_argument('--days-ahead', type=int, default=5,
                                help='Number of days ahead to predict')
    predict_parser.add_argument('--interactive', action='store_true',
                                help='Run in interactive mode')

    # Parse arguments
    args = parser.parse_args()

    # If no command is provided, print help
    if not args.command:
        parser.print_help()
        return

    # Execute the appropriate command
    if args.command == 'service':
        print("Starting scheduled service...")
        # Run data collection immediately
        collect_and_process_all_data()
        # Train models immediately
        train_models_for_all_tickers()
        # Start the scheduled job runner in a separate thread
        thread = threading.Thread(target=run_scheduled_jobs)
        thread.daemon = True
        thread.start()

        # Keep the main thread running to handle keyboard interrupts
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Service stopped by user")

    elif args.command == 'collect':
        print("Running data collection...")
        collect_and_process_all_data()
        print("Data collection completed")

    elif args.command == 'train':
        ticker = args.ticker or config.DEFAULT_TICKER
        print(f"Training models for {ticker}...")
        train_models(
            ticker=ticker,
            days_back=args.days_back,
            save_models=True
        )
        print("Model training completed")

    elif args.command == 'predict':
        if args.interactive:
            print("Running interactive prediction...")
            interactive_predict_stock()
        else:
            tickers = args.ticker if args.ticker else None  # None will use all tracked tickers
            print(f"Making predictions for {tickers if tickers else 'all tracked tickers'}...")
            results = run_predictions(tickers, args.days_ahead)

            # Print summary of predictions
            print("\nPrediction Summary:")
            for ticker, prediction in results.items():
                print(f"\n{ticker}:")
                for date, values in prediction['predictions'].items():
                    price = values['predicted_price']
                    change = values['predicted_return'] * 100
                    print(f"  {date}: ${price:.2f} ({change:+.2f}%)")

            print("\nPredictions saved to MongoDB")


if __name__ == "__main__":
    main()