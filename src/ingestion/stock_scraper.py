import yfinance as yf
import json
import os

def stringify_keys(obj):
    if isinstance(obj, dict):
        return {str(k): stringify_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [stringify_keys(i) for i in obj]
    else:
        return obj

def collect_stock_data():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
    all_stock_data = {}

    for ticker in tickers:
        stock_data = yf.download(ticker, period="max", interval="1d", auto_adjust=True)
        stock_data['volatility'] = stock_data['High'] - stock_data['Low']

        # Clean column names: remove unnecessary characters like empty string
        stock_data.columns = [col[0] if isinstance(col, tuple) else col for col in stock_data.columns]

        stock_data = stock_data.reset_index()
        stock_data_dict = stock_data.to_dict(orient='records')

        # Fix: ensure all keys are strings
        stock_data_dict = stringify_keys(stock_data_dict)

        # Save individual ticker
        output_path = os.path.join(project_root, 'data', 'raw', f'{ticker}_stock_data.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stock_data_dict, f, indent=4, default=str)

        all_stock_data[ticker] = stock_data_dict
        print(f"Collected stock data for {ticker}")

    # Save combined data
    all_output_path = os.path.join(project_root, 'data', 'raw', 'all_stock_data.json')
    os.makedirs(os.path.dirname(all_output_path), exist_ok=True)
    with open(all_output_path, 'w') as f:
        json.dump(stringify_keys(all_stock_data), f, indent=4, default=str)

    print(f"Collected stock data for multiple tickers.")
