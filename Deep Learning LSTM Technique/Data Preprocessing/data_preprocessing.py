
#
#
# The script below encompasses essential data preprocessing steps, including the
# alignment of dates, calculation of log returns, and creation of lagged columns. 
# These preparatory measures are crucial for the subsequent implementation of deep 
# learning trading strategies.
#
# aiquants Research
# (c) Dayton Nyamai
#
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# Load historical data
url = 'https://raw.githubusercontent.com/dayton-nyamai/MarketDLModels/main/data/data.csv'
raw = pd.read_csv(url, index_col=0, parse_dates=True).dropna()

symbol = 'EUR='
data = pd.DataFrame(raw[symbol])

# Align dates and rename the column containing the price data to 'price'
data = data.loc[data.index.min():data.index.max()]
data.rename(columns={symbol: 'price'}, inplace=True)

# Calculate log returns and create direction column
data['returns'] = np.log(data['price'] / data['price'].shift(1))
data.dropna(inplace=True)
data['direction'] = np.where(data['returns'] > 0, 1, -1)
data.round(4).head()

# Histogram providing a visual representation of the distribution of EUR log returns
data['returns'].hist(bins=35, figsize=(10, 6))
plt.figtext(0.5, -0.01, 'Fig. 1.1 A histogram showing the distribution of EUR log returns',
            style='italic', ha='center')
plt.show()


# Create lagged columns
lags = 8
cols = []
for lag in range(1, lags + 1):
    col = f'lag_{lag}'
    data[col] = data['returns'].shift(lag)
    cols.append(col)

# Scatter plot based on features and labels data
data.plot.scatter(x='lag_1', y='lag_2', c='returns',
                  cmap='coolwarm', figsize=(10, 6), colorbar=True)
plt.axvline(0, c='r', ls='--')
plt.axhline(0, c='r', ls='--')
plt.figtext(0.4, -0.03, 'Fig. 1.2 Scatter plot based on features and labels data',
            style='italic', ha='center')
plt.show()
