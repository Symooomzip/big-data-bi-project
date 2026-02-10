# 📊 Big Data & Business Intelligence Project

### Real-Time Data Collection, Sentiment Analysis & PowerBI Dashboards

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PowerBI](https://img.shields.io/badge/PowerBI-Dashboard-yellow?logo=powerbi)
![MongoDB](https://img.shields.io/badge/MongoDB-Database-green?logo=mongodb)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikitlearn)

## 🎯 Project Overview

A comprehensive Big Data and Business Intelligence system that collects, processes, analyzes, and visualizes real-time data from multiple sources including social media, financial markets, and news outlets. The project combines data engineering, machine learning, and business intelligence to provide actionable insights.

### Key Components

1. **📡 Data Collection Pipeline** - Real-time scraping from Twitter, Reddit, News, Stock Markets, Economic indicators
2. **🧹 Data Processing** - Automated cleaning, transformation, and validation
3. **🤖 Machine Learning** - Sentiment analysis and predictive models
4. **💾 Database Management** - MongoDB for scalable data storage
5. **📊 Business Intelligence** - Interactive PowerBI dashboards
6. **⏰ Automation** - Scheduled data collection and processing

## ✨ Features

### Data Collection

- 🐦 **Twitter Scraper**: Real-time tweets collection using Twikit
- 🔴 **Reddit Scraper**: Posts and comments from relevant subreddits
- 📰 **News Scraper**: Articles from major news sources
- 📈 **Stock Data**: Real-time stock prices (AAPL, GOOGL, MSFT, SPY)
- 💹 **Economic Indicators**: Macroeconomic data collection

### Data Processing

- ✅ Data cleaning and validation
- 🔄 Data transformation and normalization
- 📊 Feature engineering
- 💾 MongoDB integration

### Machine Learning

- 😊 Sentiment analysis (Positive/Negative/Neutral)
- 📉 Predictive modeling with XGBoost
- 🎯 Classification models
- 📈 Time series forecasting

### Business Intelligence

- 📊 Interactive PowerBI dashboards
- 📈 Real-time data visualization
- 🎨 Custom reports and analytics
- 📉 Trend analysis

## 🏗️ Project Structure

```
Big-Data-BI-Project/
├── datacol/                          # Data Collection Module
│   ├── config/                       # Configuration files
│   │   ├── db_config.json
│   │   ├── news_config.json
│   │   ├── reddit_config.json
│   │   ├── stock_config.json
│   │   └── twitter_config.json
│   ├── data/                         # Data storage
│   │   ├── raw/                      # Raw collected data
│   │   └── cleaned/                  # Processed data
│   ├── src/                          # Source code
│   │   ├── data_collection/          # Scrapers
│   │   │   ├── twitter_scraper.py
│   │   │   ├── reddit_scraper.py
│   │   │   ├── news_scraper.py
│   │   │   ├── stock_scraper.py
│   │   │   └── economic_scraper.py
│   │   ├── data_processing/          # Data cleaning
│   │   │   └── clean_data.py
│   │   ├── machine_learning/         # ML models
│   │   │   ├── models.py
│   │   │   ├── train.py
│   │   │   ├── predict.py
│   │   │   └── preprocessing.py
│   │   ├── mongodb/                  # Database operations
│   │   ├── automation/               # Scheduling
│   │   │   └── schedule_collection.py
│   │   └── main.py                   # Main entry point
│   └── requirements.txt              # Python dependencies
├── Project/                          # PowerBI & Reports
│   ├── POWERBI+DATASETS/
│   │   ├── Project_BI(Fakir_Hamouch).pbix
│   │   ├── economic.csv
│   │   ├── news.csv
│   │   ├── reddit.csv
│   │   ├── stock.csv
│   │   └── twitter.csv
│   ├── M1DS-PRESENTATION-(HAMOUCH_FAKIR).pdf
│   └── M1DS-RAPPORT-POWERBI(HAMOUCH_FAKIR).pdf
└── documents/                        # Project documentation
    ├── M1DS-BI_Projet(FAKIR_HAMOUCH).pdf
    ├── M1DS-BigDATA-Projet(HAMOUCH_FAKIR).pdf
    └── M1DS-REPORT-TraitementMultimedias(HAMOUCH_FAKIR).pdf
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- MongoDB (local or cloud instance)
- PowerBI Desktop (for viewing dashboards)
- API Keys for:
  - Twitter API
  - Reddit API
  - News API (optional)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Symooomzip/big-data-bi-project.git
cd big-data-bi-project
```

### Step 2: Set Up Virtual Environment

```bash
# Navigate to datacol directory
cd datacol

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys

Edit the configuration files in `config/` directory:

**config/twitter_config.json**

```json
{
  "api_key": "YOUR_TWITTER_API_KEY",
  "api_secret": "YOUR_TWITTER_API_SECRET",
  "access_token": "YOUR_ACCESS_TOKEN",
  "access_token_secret": "YOUR_ACCESS_TOKEN_SECRET"
}
```

**config/reddit_config.json**

```json
{
  "client_id": "YOUR_REDDIT_CLIENT_ID",
  "client_secret": "YOUR_REDDIT_CLIENT_SECRET",
  "user_agent": "YOUR_USER_AGENT"
}
```

**config/db_config.json**

```json
{
  "mongodb_uri": "mongodb://localhost:27017/",
  "database_name": "big_data_bi"
}
```

### Step 5: Set Up MongoDB

```bash
# Start MongoDB service
# Windows:
net start MongoDB
# Linux:
sudo systemctl start mongod
```

## 💻 Usage

### Data Collection

#### Collect All Data Sources

```bash
cd datacol/src
python main.py
```

#### Collect Specific Data Source

```python
# Twitter data
python data_collection/twitter_scraper.py

# Reddit data
python data_collection/reddit_scraper.py

# Stock data
python data_collection/stock_scraper.py

# News data
python data_collection/news_scraper.py

# Economic data
python data_collection/economic_scraper.py
```

### Data Processing

```bash
# Clean collected data
python data_processing/clean_data.py
```

### Machine Learning

```bash
# Train models
python machine_learning/train.py

# Make predictions
python machine_learning/predict.py
```

### Automated Data Collection

```bash
# Run scheduled collection
python automation/schedule_collection.py
```

### PowerBI Dashboard

1. Open PowerBI Desktop
2. Navigate to `Project/POWERBI+DATASETS/`
3. Open `Project_BI(Fakir_Hamouch).pbix`
4. Refresh data sources if needed

## 📊 Data Sources

### 1. Twitter Data

- Real-time tweets
- User mentions
- Hashtags
- Sentiment indicators

### 2. Reddit Data

- Posts from finance subreddits
- Comments and discussions
- Upvotes/downvotes
- Community sentiment

### 3. News Articles

- Financial news
- Market analysis
- Company announcements
- Economic reports

### 4. Stock Market Data

- **Stocks**: AAPL, GOOGL, MSFT, SPY
- **Metrics**: Open, High, Low, Close, Volume
- **Frequency**: Real-time/Daily

### 5. Economic Indicators

- GDP data
- Inflation rates
- Employment statistics
- Interest rates

## 🤖 Machine Learning Models

### Sentiment Analysis

- **Algorithm**: Scikit-learn classifiers
- **Features**: Text preprocessing, TF-IDF vectorization
- **Output**: Positive, Negative, Neutral

### Predictive Models

- **Algorithm**: XGBoost
- **Task**: Stock price prediction, trend forecasting
- **Features**: Historical prices, sentiment scores, volume

### Model Performance

- Accuracy metrics tracked in logs
- Cross-validation for robustness
- Regular model retraining

## 📈 PowerBI Dashboards

The PowerBI dashboard includes:

- 📊 **Overview Dashboard**: Key metrics and KPIs
- 📈 **Stock Analysis**: Price trends and predictions
- 😊 **Sentiment Analysis**: Social media sentiment tracking
- 📰 **News Impact**: News correlation with market movements
- 🔄 **Real-time Updates**: Live data refresh

## 🔧 Configuration

### Scheduling Data Collection

Edit `automation/schedule_collection.py` to set collection frequency:

```python
# Collect every hour
schedule.every().hour.do(collect_all_data)

# Collect every day at specific time
schedule.every().day.at("09:00").do(collect_all_data)
```

### Database Configuration

MongoDB connection settings in `config/db_config.json`:

- Connection URI
- Database name
- Collection names

## 📝 API Documentation

### Data Collection Functions

```python
from src.data_collection import twitter_scraper

# Collect tweets
tweets = twitter_scraper.collect_tweets(query="stock market", count=100)

# Save to MongoDB
twitter_scraper.save_to_db(tweets)
```

### Data Processing Functions

```python
from src.data_processing import clean_data

# Clean raw data
cleaned_data = clean_data.process_twitter_data(raw_data)
```

### Machine Learning Functions

```python
from src.machine_learning import predict

# Make prediction
prediction = predict.predict_sentiment(text)
```

## 🐛 Troubleshooting

### MongoDB Connection Issues

```bash
# Check MongoDB status
mongod --version

# Verify connection
mongo --eval "db.adminCommand('ping')"
```

### API Rate Limits

- Twitter: 450 requests per 15 minutes
- Reddit: 60 requests per minute
- Implement rate limiting in scrapers

### PowerBI Data Refresh

- Ensure CSV files are in correct location
- Update data source paths in PowerBI
- Check file permissions

## 📚 Dependencies

### Core Libraries

- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `pymongo` - MongoDB driver
- `scikit-learn` - Machine learning
- `xgboost` - Gradient boosting
- `matplotlib`, `seaborn` - Visualization

### Data Collection

- `twikit` - Twitter scraping
- `praw` - Reddit API
- `beautifulsoup4` - Web scraping
- `yfinance` - Stock data
- `requests` - HTTP requests

### Automation

- `schedule` - Task scheduling
- `streamlit` - Web dashboard (optional)

## 📄 Documentation

Detailed documentation available in the `documents/` directory:

- Project proposal
- Technical report
- PowerBI report
- Presentation slides

## 🎓 Academic Context

**Course**: M1 Data Science
**Module**: Business Intelligence & Big Data
**Authors**: HAMOUCH & FAKIR
**Institution**: [Your University]

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Data sources: Twitter, Reddit, Yahoo Finance
- Tools: PowerBI, MongoDB, Scikit-learn
- Libraries: See requirements.txt

## 👥 Authors

**HAMOUCH & FAKIR**

- GitHub: [@Symooomzip](https://github.com/Symooomzip)

## 📧 Contact

For questions or collaboration:

- Email: [mr.fakir.mohammed@gmail.com]
- GitHub Issues: [Project Issues](https://github.com/Symooomzip/big-data-bi-project/issues)

---

**Built with 💙 for Data Science & Business Intelligence**
