#Time Series Analysis
# Import necessary libraries
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Group by date to count the number of articles published per day
daily_articles = data.groupby('date').size()

# Decompose the time series into trend, seasonal, and residual components
decomposition = seasonal_decompose(daily_articles, model='additive', period=30)

# Plot the decomposition
plt.figure(figsize=(14, 10))
plt.subplot(411)
plt.plot(daily_articles, label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(decomposition.resid, label='Residual')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
