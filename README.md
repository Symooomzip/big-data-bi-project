# 📊 Near Real-Time Financial Sentiment Engine & Trading Signal Pipeline

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![MongoDB](https://img.shields.io/badge/MongoDB-NoSQL-green?logo=mongodb)
![XGBoost](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange?logo=scikitlearn)
![PowerBI](https://img.shields.io/badge/PowerBI-Dashboard-yellow?logo=powerbi)

## 🎯 Executive Summary

An end-to-end Data Engineering and Quantitative Machine Learning pipeline designed to quantify the impact of social media sentiment on short-term market movements.

Instead of relying on static datasets, this system uses a micro-batch ingestion engine to aggregate unstructured text (Twitter, Reddit, Financial News) alongside OHLCV market data. The core objective is to move beyond static dashboards and build the foundation for an event-driven decision support system capable of generating actionable trading signals.

## 🏗️ Technical Architecture

This project is structured using industry-standard decoupling principles:

```text
[Data Sources] (Twitter, Reddit, News, Yahoo Finance) 
      │
      ▼
[Ingestion Layer] ──────► Python Extractors (Micro-batch scraping pods)
      │
      ▼
[Storage Layer] ────────► MongoDB (NoSQL Raw Data Lake)
      │
      ▼
[Processing & ML] ──────► NLP Sentiment Classification & XGBoost Predictor
      │
      ▼
[Serving Layer] ────────► Power BI (Dynamic Visualizations & Alerts)
```

## 🧠 Quantitative Machine Learning Implementation

*   **Domain-Specific NLP:** Moving beyond basic bag-of-words by implementing financial text classification to accurately capture the nuanced sentiment of market-moving news and retail trading slang.
*   **Feature Engineering:** Calculated time-series features including *sentiment momentum*, lagged volatility, and volume spikes. 
*   **Predictive Modeling:** Trained an **XGBoost** model to forecast market volatility. We enforce strict awareness of time-series validation principles to prevent look-ahead bias and data leakage.
*   **Key Insight:** Predicting price direction based solely on text sentiment is notoriously difficult, but using sentiment *momentum* combined with volume spikes proved to be a highly effective volatility indicator.

## 📂 Repository Structure

The codebase follows a clean, production-ready architecture:

```text
Prjet/
├── src/                    # Source Code
│   ├── ingestion/          # Data collection scripts (Twitter, Reddit, Stock API)
│   ├── processing/         # Data cleaning and normalization logic
│   ├── models/             # NLP inference and XGBoost training scripts
│   ├── database/           # MongoDB connection handlers
│   ├── orchestration/      # Task scheduling and automation
│   └── main.py             # System entry point
├── config/                 # Environment configurations & API keys
├── data/                   # Raw & Cleaned datasets (gitignored)
├── dashboards/             # PowerBI .pbix files
├── docs/                   # Academic presentations and reports
└── requirements.txt        # Python dependencies
```

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Symooomzip/big-data-bi-project.git
   cd big-data-bi-project
   ```

2. **Install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure the environment:**
   Ensure your API keys (Twitter, Reddit) and MongoDB connection string are set in the `config/` directory.

4. **Run the pipeline:**
   ```bash
   python src/main.py
   ```

## 👥 Authors & Architecture Team

This system was architected and developed by:

*   **Mohammed Fakir** - [GitHub: @Symooomzip](https://github.com/Symooomzip) | [Email](mailto:mr.fakir.mohammed@gmail.com)
*   **Lubabah Hamouch** - [GitHub: @Lubabah-Hamouch](https://github.com/Lubabah-Hamouch)

---
*Built for the Université Mundiapolis M1 Data Science Program - Business Intelligence & Big Data Module.*
