#Exploratory Data Analysis (EDA)
import pandas as pd


# Basic statistics for headline lengths
data = pd.read_csv('D:/AI_Matery_10Acadamy/week1 10acadamy/raw_analyst_ratings.csv')
data['headline_length'] = data['headline'].apply(len)
print(data['headline_length'].describe())

# Count the number of articles per publisher
publisher_counts = data['publisher'].value_counts()
print(publisher_counts)

# Analyze publication dates
data['date'] = pd.to_datetime(data['date'])
print(data['date'].describe())

# Visualize the number of articles over time
data['date'].value_counts().sort_index().plot(kind='line', title='Articles Over Time')
