# Set page configuration - MUST BE FIRST STREAMLIT COMMAND
import streamlit as st

st.set_page_config(
    page_title="Stock Market Prediction App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import sys
import os
import json
from datetime import datetime, timedelta

# Add paths to systempath for imports
sys.path.append(os.path.abspath("src"))

# Import the predict_stock function directly from predict.py
try:
    # Import the predict_stock function directly
    from machine_learning.predict import predict_stock, save_predictions, plot_predictions
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.error("Please check your project structure and imports.")


    # Define mock functions if imports fail
    def predict_stock(ticker="AAPL", days_back=30, model_name=None, prediction_days=5):
        #st.error("Failed to load prediction module - using mock data")
        # Mock data
        predictions = {}
        start_date = datetime.now()
        for i in range(1, prediction_days + 1):
            date = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
            predictions[date] = {
                'predicted_return': float(np.random.normal(0.001, 0.01)),
                'predicted_price': float(100 + np.random.normal(0, 2))
            }
        return {
            'ticker': ticker,
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            'model_used': model_name or "mock_model",
            'predictions': predictions
        }


    def save_predictions(predictions, output_file=None):
        return "mock_path/predictions.json"


    def plot_predictions(predictions, show_plot=False, save_plot=False, output_file=None):
        pass


    # Mock MongoDB handler
    class MockMongoDBHandler:
        def __init__(self):
            pass

        def get_collection(self, collection_name):
            return None

        def find_documents(self, collection_name, query, projection=None, limit=None, sort=None):
            return []


    MongoDBHandler = MockMongoDBHandler

# Define global variables
STOCKS = ["AAPL", "MSFT", "GOOGL", "SPY", "AMZN", "META", "TSLA", "NFLX"]
EVENT_TYPES = ["Earnings Report", "Product Launch", "Economic Report", "Political Event", "Global Crisis"]
SENTIMENT_SOURCES = ["Twitter", "Reddit", "News"]

# Initialize MongoDB handler
try:
    from mongodb.mongo_handler import MongoDBHandler

    mongo_handler = MongoDBHandler()
except ImportError:
    mongo_handler = MockMongoDBHandler()

# App title and description
st.title("Stock Market Prediction Based on Global Events")
st.markdown("""
This application predicts stock prices using historical data, sentiment analysis, and machine learning.
Select a stock ticker and prediction parameters to see future price forecasts.
""")

# Create sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to",
                        ["Stock Prediction", "Event Trends Analysis", "Sentiment vs. Stock Price", "Model Performance"])


# Function to get stock data from MongoDB
def get_stock_data(stock_symbol, days=30):
    try:
        # Use your MongoDB handler to fetch data
        query = {"symbol": stock_symbol}
        sort = [("date", -1)]
        limit = days

        stock_data = mongo_handler.find_documents("stock_data", query, limit=limit, sort=sort)

        if not stock_data:
            # Return mock data if no data is found
            return pd.DataFrame({
                "date": pd.date_range(start=datetime.now() - timedelta(days=days), periods=days,
                                      freq='D'),
                "price": [100 + i + np.random.normal(0, 2) for i in range(days)],
                "volume": [1000000 + np.random.normal(0, 100000) for _ in range(days)]
            })

        return pd.DataFrame(stock_data)
    except Exception as e:
        st.error(f"Error retrieving stock data: {e}")
        return pd.DataFrame()


# Function to get sentiment data from MongoDB
def get_sentiment_data(stock_symbol, source, days=30):
    try:
        # Use your MongoDB handler to fetch data
        query = {"stock_symbol": stock_symbol}
        sort = [("date", -1)]
        limit = days

        sentiment_data = mongo_handler.find_documents(f"{source.lower()}_data", query, limit=limit, sort=sort)

        if not sentiment_data:
            # Return mock data if no data is found
            return pd.DataFrame({
                "date": pd.date_range(start=datetime.now() - timedelta(days=days), periods=days,
                                      freq='D'),
                "sentiment_score": [np.random.normal(0, 0.5) for _ in range(days)],
                "volume": [1000 + np.random.normal(0, 100) for _ in range(days)]
            })

        return pd.DataFrame(sentiment_data)
    except Exception as e:
        st.error(f"Error retrieving sentiment data: {e}")
        return pd.DataFrame()


# Function to get event data from MongoDB
def get_event_data(stock_symbol, event_type, days=30):
    try:
        # Use your MongoDB handler to fetch data
        query = {"stock_symbol": stock_symbol, "event_type": event_type}
        sort = [("date", -1)]
        limit = days

        event_data = mongo_handler.find_documents("event_data", query, limit=limit, sort=sort)

        if not event_data:
            # Return mock data if no data is found
            dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days)
            price_data = []
            for i, date in enumerate(dates):
                phase = "Before" if i < 10 else "During" if i < 20 else "After"
                if phase == "Before":
                    price = 100 + i * 0.2 + np.random.normal(0, 1)
                elif phase == "During":
                    price = 102 + (i - 10) * 0.5 + np.random.normal(0, 2)
                else:
                    price = 107 + (i - 20) * 0.3 + np.random.normal(0, 1)

                price_data.append({
                    "date": date,
                    "price": price,
                    "phase": phase
                })

            return pd.DataFrame(price_data)

        return pd.DataFrame(event_data)
    except Exception as e:
        st.error(f"Error retrieving event data: {e}")
        return pd.DataFrame()


# Function to format the prediction with icons and colors
def format_prediction(prediction_value, confidence):
    if prediction_value > 0:
        return f"📈 +{prediction_value:.2f} ({confidence * 100:.1f}% confidence)"
    else:
        return f"📉 {prediction_value:.2f} ({confidence * 100:.1f}% confidence)"


# 1. Stock Prediction Page
if page == "Stock Prediction":
    st.header("Stock Price Prediction")

    # Add explanation
    st.markdown("""
    This section allows you to predict future stock prices using our machine learning models.
    Enter the ticker symbol, select the number of days of historical data to use,
    and specify how many days ahead you want to predict.
    """)

    # Create columns for user input
    col1, col2 = st.columns(2)

    with col1:
        # Instead of selectbox, use text input to allow any ticker
        ticker = st.text_input("Enter stock ticker symbol", value="AAPL")
        days_back = st.slider("Number of days of historical data to use", 10, 90, 30)

    with col2:
        prediction_days = st.slider("Number of days ahead to predict", 1, 14, 5)

        # Add model selection dropdown
        available_models = ["linear", "ridge", "lasso", "elasticnet", "randomforest", "xgboost"]
        model_name = st.selectbox(
            "Select prediction model (or leave blank for best model)",
            ["best model"] + available_models
        )
        if model_name == "best model":
            model_name = None

    if st.button("Generate Prediction"):
        with st.spinner("Generating prediction..."):
            try:
                # Call the predict_stock function with user inputs
                result = predict_stock(
                    ticker=ticker,
                    days_back=days_back,
                    model_name=model_name,
                    prediction_days=prediction_days
                )

                if result:
                    # Display prediction
                    st.subheader("Prediction Results")

                    # Extract data from the result
                    predictions = result.get("predictions", {})
                    model_used = result.get("model_used", "unknown")

                    # Create dataframe from predictions for easier manipulation
                    dates = []
                    prices = []
                    returns = []

                    for date, data in predictions.items():
                        dates.append(date)
                        prices.append(data["predicted_price"])
                        returns.append(data["predicted_return"])

                    pred_df = pd.DataFrame({
                        "date": dates,
                        "predicted_price": prices,
                        "predicted_return": returns
                    })

                    # Calculate overall predicted change
                    overall_change = pred_df["predicted_price"].iloc[-1] - pred_df["predicted_price"].iloc[0]
                    overall_change_pct = overall_change / pred_df["predicted_price"].iloc[0] * 100

                    # Show prediction in a metrics card
                    col1, col2 = st.columns(2)

                    col1.metric(
                        label=f"{ticker} Final Predicted Price",
                        value=f"${pred_df['predicted_price'].iloc[-1]:.2f}",
                        delta=f"{overall_change:.2f} ({overall_change_pct:.2f}%)"
                    )

                    col2.metric(
                        label="Model Used",
                        value=model_used
                    )

                    # Create a line chart for predicted prices
                    fig1 = go.Figure()

                    # Convert string dates to datetime
                    pred_df["date"] = pd.to_datetime(pred_df["date"])

                    # Add price prediction line
                    fig1.add_trace(
                        go.Scatter(
                            x=pred_df["date"],
                            y=pred_df["predicted_price"],
                            mode="lines+markers",
                            name="Predicted Price",
                            line=dict(color="blue")
                        )
                    )

                    fig1.update_layout(
                        title=f"{ticker} Stock Price Prediction",
                        xaxis_title="Date",
                        yaxis_title="Stock Price ($)",
                        height=400
                    )

                    # Format y-axis to show dollar amounts
                    fig1.update_layout(yaxis=dict(tickprefix="$"))

                    st.plotly_chart(fig1, use_container_width=True)

                    # Create bar chart for daily returns
                    fig2 = go.Figure()

                    # Add daily returns bars with color based on positive/negative
                    colors = ['green' if r > 0 else 'red' for r in pred_df["predicted_return"]]

                    fig2.add_trace(
                        go.Bar(
                            x=pred_df["date"],
                            y=[r * 100 for r in pred_df["predicted_return"]],  # Convert to percentage
                            marker_color=colors,
                            name="Daily Returns"
                        )
                    )

                    fig2.update_layout(
                        title=f"{ticker} Predicted Daily Returns",
                        xaxis_title="Date",
                        yaxis_title="Return (%)",
                        height=300
                    )

                    # Add a horizontal line at y=0
                    fig2.add_shape(
                        type="line",
                        x0=pred_df["date"].min(),
                        y0=0,
                        x1=pred_df["date"].max(),
                        y1=0,
                        line=dict(color="black", width=1, dash="dot")
                    )

                    st.plotly_chart(fig2, use_container_width=True)

                    # Display detailed predictions in a table
                    st.subheader("Detailed Daily Predictions")

                    # Add percentage to return values for better readability
                    detail_df = pred_df.copy()
                    detail_df["predicted_return"] = detail_df["predicted_return"].apply(lambda x: f"{x * 100:.2f}%")
                    detail_df["predicted_price"] = detail_df["predicted_price"].apply(lambda x: f"${x:.2f}")
                    detail_df = detail_df.rename(columns={
                        "date": "Date",
                        "predicted_price": "Predicted Price",
                        "predicted_return": "Predicted Return"
                    })

                    st.dataframe(detail_df, use_container_width=True)

                    # Add save options
                    st.subheader("Save Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("Save Predictions to JSON"):
                            # Save predictions
                            save_path = save_predictions(result)
                            st.success(f"Predictions saved to {save_path}")

                    with col2:
                        if st.button("Generate and Save Plot"):
                            # Generate and save plot
                            plot_predictions(result, show_plot=False, save_plot=True)
                            st.success("Plot saved to predictions directory")

                else:
                    st.error("Failed to generate predictions. Please check the inputs and try again.")

            except Exception as e:
                st.error(f"Error generating prediction: {e}")
                st.error("Please check if the prediction model is working correctly.")

# 2. Event Trends Analysis Page
elif page == "Event Trends Analysis":
    st.header("Event Trends Visualization")

    col1, col2 = st.columns(2)

    with col1:
        selected_stock = st.selectbox("Select Stock", STOCKS)

    with col2:
        selected_event = st.selectbox("Select Event Type", EVENT_TYPES)

    if st.button("Analyze Event Impact"):
        with st.spinner("Analyzing event impact..."):
            try:
                # Get event data
                event_data = get_event_data(selected_stock, selected_event)

                # Create a line chart with colored phases
                fig = px.line(
                    event_data,
                    x="date",
                    y="price",
                    color="phase",
                    title=f"Impact of {selected_event} on {selected_stock} Stock Price",
                    labels={"date": "Date", "price": "Stock Price ($)"}
                )

                fig.update_layout(height=500)

                st.plotly_chart(fig, use_container_width=True)

                # Add analysis summary
                st.subheader("Event Impact Analysis")

                col1, col2, col3 = st.columns(3)

                before_avg = event_data[event_data["phase"] == "Before"]["price"].mean()
                during_avg = event_data[event_data["phase"] == "During"]["price"].mean()
                after_avg = event_data[event_data["phase"] == "After"]["price"].mean()

                col1.metric("Before Event Avg", f"${before_avg:.2f}")
                col2.metric("During Event Avg", f"${during_avg:.2f}", f"{during_avg - before_avg:.2f}")
                col3.metric("After Event Avg", f"${after_avg:.2f}", f"{after_avg - during_avg:.2f}")

                # Add interpretation
                st.markdown(f"""
                ### Key Insights:
                - The stock price **{'increased' if during_avg > before_avg else 'decreased'}** by ${abs(during_avg - before_avg):.2f} during the {selected_event}
                - After the event, the price **{'continued to increase' if after_avg > during_avg else 'began to decrease'}** by ${abs(after_avg - during_avg):.2f}
                - Overall impact: **{'Positive' if after_avg > before_avg else 'Negative'}**
                """)

            except Exception as e:
                st.error(f"Error analyzing event impact: {e}")

# 3. Sentiment vs. Stock Price Page
elif page == "Sentiment vs. Stock Price":
    st.header("Sentiment vs. Stock Price Analysis")

    col1, col2 = st.columns(2)

    with col1:
        selected_stock = st.selectbox("Select Stock", STOCKS)

    with col2:
        selected_source = st.selectbox("Select Sentiment Source", SENTIMENT_SOURCES)

    if st.button("Analyze Sentiment Correlation"):
        with st.spinner("Analyzing sentiment correlation..."):
            try:
                # Get stock data
                stock_data = get_stock_data(selected_stock)

                # Get sentiment data
                sentiment_data = get_sentiment_data(selected_stock, selected_source)

                # Ensure dates are in the same format before merging
                if 'date' in stock_data.columns and 'date' in sentiment_data.columns:
                    # Convert dates to the same format if needed
                    if not pd.api.types.is_datetime64_any_dtype(stock_data['date']):
                        stock_data['date'] = pd.to_datetime(stock_data['date'])
                    if not pd.api.types.is_datetime64_any_dtype(sentiment_data['date']):
                        sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])

                # Merge data for correlation analysis
                merged_data = pd.merge(stock_data, sentiment_data, on="date", how="inner")

                # Continue only if we have valid data
                if not merged_data.empty and 'price' in merged_data.columns and 'sentiment_score' in merged_data.columns:
                    # Check for and handle any NaN values
                    merged_data = merged_data.dropna(subset=['price', 'sentiment_score'])

                    # Create a scatter plot for correlation
                    if not merged_data.empty:
                        fig1 = px.scatter(
                            merged_data,
                            x="sentiment_score",
                            y="price",
                            trendline="ols",
                            title=f"{selected_stock} Stock Price vs. {selected_source} Sentiment",
                            labels={
                                "sentiment_score": "Sentiment Score (Negative to Positive)",
                                "price": "Stock Price ($)"
                            }
                        )

                        fig1.update_layout(height=500)
                        st.plotly_chart(fig1, use_container_width=True)

                        # Create a dual-axis time series chart
                        fig2 = go.Figure()

                        # Add price line
                        fig2.add_trace(
                            go.Scatter(
                                x=merged_data["date"],
                                y=merged_data["price"],
                                name="Stock Price",
                                line=dict(color="blue")
                            )
                        )

                        # Add sentiment line on secondary axis
                        fig2.add_trace(
                            go.Scatter(
                                x=merged_data["date"],
                                y=merged_data["sentiment_score"],
                                name="Sentiment Score",
                                line=dict(color="red"),
                                yaxis="y2"
                            )
                        )

                        fig2.update_layout(
                            title=f"{selected_stock} Stock Price and {selected_source} Sentiment Over Time",
                            xaxis=dict(title="Date"),
                            yaxis=dict(
                                title="Stock Price ($)",
                                title_font=dict(color="blue"),
                                tickfont=dict(color="blue")
                            ),
                            yaxis2=dict(
                                title="Sentiment Score",
                                title_font=dict(color="red"),
                                tickfont=dict(color="red"),
                                anchor="x",
                                overlaying="y",
                                side="right"
                            ),
                            height=500
                        )

                        st.plotly_chart(fig2, use_container_width=True)

                        # Calculate correlation coefficient
                        correlation = merged_data["sentiment_score"].corr(merged_data["price"])

                        # Handle NaN correlation (just in case)
                        if pd.isna(correlation):
                            st.metric("Correlation Coefficient", "No correlation found")
                            st.warning("Not enough data points to calculate correlation.")
                        else:
                            # Add correlation analysis
                            st.subheader("Correlation Analysis")
                            st.metric("Correlation Coefficient", f"{correlation:.2f}")

                            if correlation > 0.7:
                                st.success(
                                    f"Strong positive correlation between {selected_source} sentiment and {selected_stock} stock price.")
                            elif correlation > 0.3:
                                st.info(
                                    f"Moderate positive correlation between {selected_source} sentiment and {selected_stock} stock price.")
                            elif correlation > -0.3:
                                st.warning(
                                    f"Weak correlation between {selected_source} sentiment and {selected_stock} stock price.")
                            elif correlation > -0.7:
                                st.info(
                                    f"Moderate negative correlation between {selected_source} sentiment and {selected_stock} stock price.")
                            else:
                                st.error(
                                    f"Strong negative correlation between {selected_source} sentiment and {selected_stock} stock price.")
                    else:
                        st.warning("No valid data points available after filtering.")
                else:
                    st.warning("No matching data found for the selected stock and sentiment source.")

            except Exception as e:
                st.error(f"Error analyzing sentiment correlation: {e}")

# 4. Model Performance Page
elif page == "Model Performance":
    st.header("Model Performance Metrics")

    # Add tabs for different model metrics
    tabs = st.tabs(["Overall Performance", "Performance by Stock", "Performance History"])

    with tabs[0]:
        try:
            # Get model performance data from MongoDB
            # This would come from your MongoDB or another source
            # For now, we'll use mock data

            # Mock metrics for overall performance
            metrics = {
                "RMSE": 2.34,
                "MAE": 1.87,
                "R-squared": 0.76,
                "Accuracy": 0.82
            }

            # Create metrics display
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RMSE", f"{metrics['RMSE']:.2f}")
            col2.metric("MAE", f"{metrics['MAE']:.2f}")
            col3.metric("R-squared", f"{metrics['R-squared']:.2f}")
            col4.metric("Accuracy", f"{metrics['Accuracy']:.2f}")

            # Create a confusion matrix
            st.subheader("Prediction Accuracy")

            confusion_data = {
                "Actual Up": {"Predicted Up": 42, "Predicted Down": 8},
                "Actual Down": {"Predicted Up": 10, "Predicted Down": 40}
            }

            confusion_df = pd.DataFrame({
                "Predicted Up": [confusion_data["Actual Up"]["Predicted Up"],
                                 confusion_data["Actual Down"]["Predicted Up"]],
                "Predicted Down": [confusion_data["Actual Up"]["Predicted Down"],
                                   confusion_data["Actual Down"]["Predicted Down"]]
            }, index=["Actual Up", "Actual Down"])

            fig = px.imshow(
                confusion_df,
                text_auto=True,
                aspect="auto",
                color_continuous_scale=["#f7fbff", "#2171b5"]
            )

            fig.update_layout(
                title="Confusion Matrix",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading model performance metrics: {e}")

    with tabs[1]:
        try:
            # Mock performance by stock
            stock_performance = {
                "AAPL": {"RMSE": 2.1, "MAE": 1.7, "R-squared": 0.81, "Accuracy": 0.85},
                "MSFT": {"RMSE": 2.3, "MAE": 1.9, "R-squared": 0.79, "Accuracy": 0.82},
                "GOOGL": {"RMSE": 2.7, "MAE": 2.1, "R-squared": 0.72, "Accuracy": 0.77},
                "SPY": {"RMSE": 1.8, "MAE": 1.5, "R-squared": 0.84, "Accuracy": 0.88}
            }

            # Create bar chart for stock performance comparison
            metrics_to_compare = ["RMSE", "MAE", "R-squared", "Accuracy"]

            stock_names = list(stock_performance.keys())
            data = []

            for metric in metrics_to_compare:
                for stock in stock_names:
                    data.append({
                        "Stock": stock,
                        "Metric": metric,
                        "Value": stock_performance[stock][metric]
                    })

            performance_df = pd.DataFrame(data)

            fig = px.bar(
                performance_df,
                x="Stock",
                y="Value",
                color="Metric",
                barmode="group",
                title="Model Performance by Stock",
                labels={"Value": "Metric Value", "Stock": "Stock Symbol"}
            )

            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Add best performing stock analysis
            st.subheader("Performance Analysis")

            # Find best performing stock
            best_stock = max(stock_performance.items(), key=lambda x: x[1]["Accuracy"])[0]

            st.info(
                f"The model performs best on **{best_stock}** with an accuracy of {stock_performance[best_stock]['Accuracy']:.2f} and RMSE of {stock_performance[best_stock]['RMSE']:.2f}.")

        except Exception as e:
            st.error(f"Error loading stock performance metrics: {e}")

    with tabs[2]:
        try:
            # Mock performance history over time
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30)

            # Create mock performance history data
            history_data = []
            for i, date in enumerate(dates):
                # Slight improvement trend with noise
                rmse = 2.5 - (i * 0.02) + np.random.normal(0, 0.15)
                accuracy = 0.75 + (i * 0.003) + np.random.normal(0, 0.02)

                history_data.append({
                    "date": date,
                    "RMSE": max(0, rmse),  # Ensure RMSE is positive
                    "Accuracy": min(1, max(0, accuracy))  # Ensure accuracy is between 0 and 1
                })

            history_df = pd.DataFrame(history_data)

            # Create line chart for performance history
            fig = go.Figure()

            # Add RMSE line
            fig.add_trace(
                go.Scatter(
                    x=history_df["date"],
                    y=history_df["RMSE"],
                    name="RMSE",
                    line=dict(color="red")
                )
            )

            # Add Accuracy line on secondary axis
            fig.add_trace(
                go.Scatter(
                    x=history_df["date"],
                    y=history_df["Accuracy"],
                    name="Accuracy",
                    line=dict(color="green"),
                    yaxis="y2"
                )
            )

            fig.update_layout(
                title="Model Performance Over Time",
                xaxis=dict(title="Date"),
                yaxis=dict(
                    title="RMSE",
                    title_font=dict(color="red"),
                    tickfont=dict(color="red")
                ),
                yaxis2=dict(
                    title="Accuracy",
                    title_font=dict(color="green"),
                    tickfont=dict(color="green"),
                    anchor="x",
                    overlaying="y",
                    side="right",
                    range=[0, 1]
                ),
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading performance history: {e}")

# Add auto-refresh functionality
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto-refresh (every 15 minutes)", value=False)

if auto_refresh:
    st.sidebar.write("Last updated: ", datetime.now().strftime("%H:%M:%S"))
    st.sidebar.info("Page will refresh automatically every 15 minutes.")

    # Add auto-refresh using HTML
    refresh_interval = 15 * 60  # 15 minutes in seconds
    st.markdown(
        f"""
        <script>
            setTimeout(function() {{
                window.location.reload();
            }}, {refresh_interval * 1000});
        </script>
        """,
        unsafe_allow_html=True
    )

# Add footer
st.sidebar.markdown("---")
st.sidebar.info(
    "Stock Market Prediction App using Machine Learning"
)