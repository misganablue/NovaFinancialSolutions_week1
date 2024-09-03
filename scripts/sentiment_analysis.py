#Sentiment Analysis of Financial News Headlines
# Import necessary libraries
# Import necessary libraries
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download('vader_lexicon')

# Load the dataset
data = pd.read_csv('D:/AI_Matery_10Acadamy/week1 10acadamy/raw_analyst_ratings.csv')

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Calculate the sentiment scores for the headlines
data['sentiment'] = data['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Categorize the sentiment scores into positive, negative, and neutral
data['sentiment_category'] = pd.cut(data['sentiment'], bins=[-1, -0.5, 0.5, 1], labels=['Negative', 'Neutral', 'Positive'])

# Display the sentiment analysis results
print(data[['headline', 'sentiment', 'sentiment_category']].head())

