#Correlation Analysis
# Import necessary libraries
import yfinance as yf

# Define the ticker and fetch the stock data
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-01-01'
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Calculate daily returns
stock_data['Daily Returns'] = stock_data['Close'].pct_change()

# Align the sentiment data with stock data based on date
combined_data = data[['date', 'sentiment']].merge(stock_data[['Close', 'Daily Returns']], left_on='date', right_index=True)

# Calculate correlation
correlation = combined_data['sentiment'].corr(combined_data['Daily Returns'])
print("Correlation between sentiment and daily stock returns:", correlation)
