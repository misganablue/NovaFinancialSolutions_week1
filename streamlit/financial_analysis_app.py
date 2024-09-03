import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from statsmodels.tsa.seasonal import seasonal_decompose
from textblob import TextBlob
import nltk
import gdown
import os
import torch

# URL for the file to be downloaded from Google Drive
file_id = 'YOUR_FILE_ID'  # Replace with your file ID
download_url = f'https://drive.google.com/file/d/1tLHusoOQOm1cU_7DtLNbykgFgJ_piIpd/view?usp=sharing'

# Path to save the downloaded CSV file
csv_file_path = 'large_dataset.csv'  # Change to your desired file path

# Function to download the CSV file
def download_csv():
    if not os.path.exists(csv_file_path):
        st.write("Downloading CSV file from Google Drive...")
        gdown.download(download_url, csv_file_path, quiet=False)
        st.write("Download completed.")
    else:
        st.write("CSV file is already downloaded.")

# Download the CSV file if not already downloaded
download_csv()

# Load the CSV file into a pandas DataFrame
@st.cache_data
def load_data():
    st.write("Loading data...")
    data = pd.read_csv(csv_file_path)
    st.write("Data loaded successfully.")
    return data

# Load data into a DataFrame
data = load_data()

# Display the data in Streamlit
st.title("Large CSV File Loader with Streamlit")
st.write("Displaying the first few rows of the dataset:")
st.write(data.head())

# Ensure necessary libraries are downloaded
nltk.download('vader_lexicon')

# Set up the Streamlit app layout
st.set_page_config(page_title="Nova Financial Solutions - Predictive Analytics", layout="wide")
st.title("Nova Financial Solutions: Predictive Analytics for Financial Forecasting")

# Sidebar for user navigation
st.sidebar.header('Navigation')
options = st.sidebar.radio("Choose a section", ['Introduction', 'Data Overview', 'Sentiment Analysis', 'Exploratory Data Analysis (EDA)', 'Time Series Analysis', 'Correlation Analysis', 'Conclusions and Recommendations'])

# Introduction Section
if options == 'Introduction':
    st.header("Project Overview")
    st.write("""
    Nova Financial Solutions aims to enhance its predictive analytics capabilities to boost financial forecasting accuracy and operational efficiency. 
    This project involves analyzing financial news headlines and correlating their sentiments with stock price movements. 
    Key components of this project include sentiment analysis, exploratory data analysis (EDA), time series analysis, and correlation analysis.
    """)
    st.image("https://www.example.com/financial_image.jpg", use_column_width=True)  # You can use a placeholder image or upload your own.

# Data Overview Section
elif options == 'Data Overview':
    st.header("Dataset Overview")
    st.write("""
    The dataset consists of financial news articles and associated metadata such as headlines, URLs, publishers, publication dates, and stock ticker symbols.
    """)
    
    # Load the dataset (example path provided, modify it as needed)
    @st.cache
    def load_data():
        data = pd.read_csv('D:/AI_Matery_10Acadamy/NovaFinancialSolutions_week1/raw_analyst_ratings.csv')
        return data
    
    data = load_data()
    st.subheader("Financial News Dataset")
    st.write(data.head(10))
    
    # Display descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(data.describe())

# Sentiment Analysis Section
elif options == 'Sentiment Analysis':
    st.header("Sentiment Analysis of Financial News Headlines")

    # Sentiment analysis using NLTK VADER
    sia = SentimentIntensityAnalyzer()
    data['sentiment'] = data['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])

    # Categorizing sentiment
    data['sentiment_category'] = pd.cut(data['sentiment'], bins=[-1, -0.5, 0.5, 1], labels=['Negative', 'Neutral', 'Positive'])
    
    st.subheader("Sentiment Scores")
    st.write(data[['headline', 'sentiment', 'sentiment_category']].head(10))

    # Sentiment distribution plot
    st.subheader("Sentiment Distribution")
    sentiment_counts = data['sentiment_category'].value_counts()
    st.bar_chart(sentiment_counts)

# Exploratory Data Analysis (EDA) Section
elif options == 'Exploratory Data Analysis (EDA)':
    st.header("Exploratory Data Analysis (EDA)")

    # Headline length analysis
    data['headline_length'] = data['headline'].apply(lambda x: len(x.split()))
    
    st.subheader("Headline Length Distribution")
    st.write("Average headline length: ", np.mean(data['headline_length']))
    st.hist_chart(data['headline_length'])

    # Articles per publisher
    st.subheader("Number of Articles per Publisher")
    publisher_counts = data['publisher'].value_counts()
    st.bar_chart(publisher_counts)

# Time Series Analysis Section
elif options == 'Time Series Analysis':
    st.header("Time Series Analysis of News Publications")

    # Parse dates and set index
    data['date'] = pd.to_datetime(data['date'])
    daily_headlines = data.groupby(data['date'].dt.date).size()
    
    st.subheader("Daily Headline Counts Over Time")
    st.line_chart(daily_headlines)

    # Perform seasonal decomposition
    st.subheader("Seasonal Decomposition")
    decomposition = seasonal_decompose(daily_headlines, model='additive', period=7)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    decomposition.observed.plot(ax=ax1, title='Observed')
    decomposition.trend.plot(ax=ax2, title='Trend')
    decomposition.seasonal.plot(ax=ax3, title='Seasonal')
    decomposition.resid.plot(ax=ax4, title='Residual')
    st.pyplot(fig)

# Correlation Analysis Section
elif options == 'Correlation Analysis':
    st.header("Correlation Analysis between Sentiment and Stock Movements")

    # Load stock price data from Yahoo Finance
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Perform correlation analysis
    data['date'] = pd.to_datetime(data['date'])
    merged_data = pd.merge(data[['date', 'sentiment']], stock_data[['Close']], left_on='date', right_index=True)
    merged_data['daily_returns'] = merged_data['Close'].pct_change()

    correlation = merged_data['sentiment'].corr(merged_data['daily_returns'])
    
    st.subheader("Correlation Result")
    st.write(f"Correlation between news sentiment and daily stock returns: {correlation:.2f}")

# Conclusions and Recommendations Section
elif options == 'Conclusions and Recommendations':
    st.header("Conclusions and Recommendations")
    st.write("""
    **Conclusions:**
    - Sentiment analysis shows a moderate relationship between news sentiment and stock price movements.
    - Technical indicators provide additional insights that can enhance trading strategies.
    
    **Recommendations:**
    - Combine sentiment analysis with technical indicators to improve predictive accuracy.
    - Focus on high-impact news events for better market predictions.
    - Explore machine learning models to refine prediction accuracy further.
    """)

    st.write("Thank you for exploring the project! For more information, please contact us.")

# Footer
st.sidebar.info("Developed by Nova Financial Solutions - Data Analytics Team")

