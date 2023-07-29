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

# Import the necessary libraries
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

from pylab import mpl, plt 
plt.style.use('seaborn-v0_8') 
mpl.rcParams['font.family'] = 'serif' 
%matplotlib inline

# Load the historical data and drop any row with missing values
url = 'https://raw.githubusercontent.com/dayton-nyamai/MarketDLModels/main/data/data.csv'
data = pd.read_csv(url, index_col=0, parse_dates=True).dropna()

symbol = ['EUR=']
data = pd.DataFrame(data[symbol]) 

# Align dates and rename the column containing the price data to 'price'
start_date = data.index.min()
end_date = data.index.max()
data = data.loc[start_date:end_date]
data.rename(columns={'EUR=': 'price'}, inplace=True)

# Calculate log returns and create direction column
data['returns'] = np.log(data['price'] / data['price'].shift(1))
data.dropna(inplace=True)
data['direction'] = np.where(data['returns'] > 0, 1, -1) 
data.round(4).head()

# Histogram providing a visual representation of the distribution of EUR log returns
data['returns'].hist(bins=35, figsize=(10, 6));
# Add figure caption
plt.figtext(0.5, -0.01, 'Fig. 1.1 A histogram  showing the distribution of EUR log returns ', style='italic',ha='center')

# Show the plot
plt.show()

# Create the features data by lagging the log returns
lags = 5

def create_lags(data): 
    global cols
    cols = []
    for lag in range(1, lags + 1):
        col = 'lag_{}'.format(lag)
        data[col] = data['returns'].shift(lag)
        cols.append(col)
create_lags(data)
data.round(4).tail()


# Scatter plot based on features and labels data
data.plot.scatter(x='lag_1', y='lag_2', c='returns', cmap='coolwarm', figsize=(10, 6), colorbar=True)

# Adding vertical and horizontal lines at 0
plt.axvline(0, c='r', ls='--')
plt.axhline(0, c='r', ls='--')

# Add figure caption
plt.figtext(0.4, -0.03, 
            'Fig. 1.2 A scatter plot based on features and labels data', 
            style='italic',ha='center')

# Show the plot
plt.show()
