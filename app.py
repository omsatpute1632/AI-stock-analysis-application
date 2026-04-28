import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px

st.set_page_config(page_title="AI Financial Intelligence Platform")

st.title("AI Financial Intelligence Platform")

menu = st.sidebar.selectbox(
    "Choose Feature",
    [
        "Stock Prediction",
        "News Sentiment Analysis",
        "Portfolio Optimization",
        "Risk Analysis"
    ]
)

# STOCK PREDICTION

st.header("Stock Price Prediction")

ticker = st.text_input("Enter Stock Ticker (Example: TSLA or TATAMOTORS)", "TSLA")

# Support Indian stocks automatically
indian_stocks = ["TATAMOTORS", "RELIANCE", "INFY", "HDFCBANK", "SBIN"]

if ticker.upper() in indian_stocks:
    ticker = ticker.upper() + ".NS"

if st.button("Run Prediction"):

    data = yf.download(ticker, start="2020-01-01")

    # Fix multi column issue
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data["Prediction"] = data["Close"].shift(-1)

    data = data.dropna()

    X = data[["Close"]]
    y = data["Prediction"]

    model = LinearRegression()
    model.fit(X, y)

    last_price = float(data["Close"].iloc[-1])

    prediction = model.predict([[last_price]])

    st.success(f"Predicted Next Price: {prediction[0]:.2f}")

    fig = px.line(data, y="Close", title="Stock Price History")
    st.plotly_chart(fig)

# Auto support Indian stocks
indian_stocks = ["TATAMOTORS", "RELIANCE", "INFY", "HDFCBANK", "SBIN"]

if ticker.upper() in indian_stocks:
    ticker = ticker.upper() + ".NS"

    if st.button("Run Prediction"):

        data = yf.download(ticker, start="2020-01-01")
        data.columns = data.columns.get_level_values(0)

        data["Prediction"] = data["Close"].shift(-1)

        data = data.dropna()

        X = data[["Close"]]
        y = data["Prediction"]

        model = LinearRegression()
        model.fit(X, y)

        last_price = float(data["Close"].iloc[-1])

        prediction = model.predict([[last_price]])

        st.success(f"Predicted Next Price: {prediction[0]:.2f}")

        fig = px.line(data, y="Close", title="Stock Price History")
        st.plotly_chart(fig)


# NEWS SENTIMENT

elif menu == "News Sentiment Analysis":

    st.header("Financial News Sentiment")

    text = st.text_area("Enter Financial News Text")

    if st.button("Analyze Sentiment"):

        sia = SentimentIntensityAnalyzer()

        score = sia.polarity_scores(text)

        st.write(score)

        if score["compound"] > 0.05:
            st.success("Positive Sentiment")
        elif score["compound"] < -0.05:
            st.error("Negative Sentiment")
        else:
            st.warning("Neutral Sentiment")


# PORTFOLIO OPTIMIZATION

elif menu == "Portfolio Optimization":

    st.header("Portfolio Optimization")

    stocks = st.text_input(
        "Enter Stocks (comma separated)",
        "AAPL,TSLA,MSFT"
    )

    if st.button("Optimize Portfolio"):

        stocks = stocks.split(",")

        data = yf.download(stocks, start="2022-01-01")["Close"]

        returns = data.pct_change().dropna()

        mean_returns = returns.mean()

        weights = np.random.random(len(stocks))
        weights /= np.sum(weights)

        portfolio_return = np.sum(mean_returns * weights)

        st.subheader("Recommended Allocation")

        allocation = pd.DataFrame({
            "Stock": stocks,
            "Weight": weights
        })

        st.dataframe(allocation)

        st.success(f"Expected Portfolio Return: {portfolio_return:.4f}")

        fig = px.line(data, title="Stock Prices")

        st.plotly_chart(fig)


# RISK ANALYSIS

elif menu == "Risk Analysis":

    st.header("Portfolio Risk Analysis")

    stocks = st.text_input(
        "Enter Portfolio Stocks",
        "AAPL,TSLA,MSFT"
    )

    if st.button("Calculate Risk"):

        stocks = stocks.split(",")

        data = yf.download(stocks, start="2022-01-01")["Close"]

        returns = data.pct_change().dropna()

        volatility = returns.std()

        st.subheader("Stock Volatility")

        st.dataframe(volatility)

        portfolio_risk = np.mean(volatility)

        st.error(f"Overall Portfolio Risk Score: {portfolio_risk:.4f}")

        fig = px.line(data, title="Portfolio Price Trends")

        st.plotly_chart(fig)