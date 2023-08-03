#
# The script below  implements the Long Short Term Memory (LSTM) model for predicting market movements.
# The script includes  essential steps such as model fitting, training, evaluation, and prediction on  
# both training and testing data. Additionally, it compares the performance of the RNNs trading strategy
# with the passive benchmark returns.
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
url = 'https://raw.githubusercontent.com/dayton-nyamai/MarketDLModels/main/data/historical_data.csv'
raw = pd.read_csv(url, index_col=0, parse_dates=True).dropna() 
raw.info() #the raw data meta information

# Select the symbol and create a DataFrame
symbol = ['EURUSD=X']
data = pd.DataFrame(raw[symbol])

# Align dates and rename the column containing the price data to 'price'.
start_date = data.index.min()
end_date = data.index.max()
data = data.loc[start_date:end_date]
data.rename(columns={'EURUSD=X': 'price'}, inplace=True)
data.round(4).head()

# Calculate log returns and create direction column
data['returns'] = np.log(data['price'] / data['price'].shift(1))
data.dropna(inplace=True)
data['direction'] = np.where(data['returns'] > 0, 1, 0)

# Create the features data by lagging the log returns
lags = 5
cols = [f'lag_{lag}' for lag in range(1, lags+1)]
for col in cols:
    data[col] = data['returns'].shift(int(col.split('_')[1]))
data.dropna(inplace=True)

# Split the data into training and test sets
cutoff = '2018-12-31'
training_data = data[data.index < cutoff].copy()
test_data = data[data.index >= cutoff].copy()

# Normalize the training and test data
mu, std = training_data.mean(), training_data.std()
training_data_ = (training_data - mu) / std
test_data_ = (test_data - mu) / std

# Reshape the training and test data for LSTM input
X_train = np.array(training_data_[cols])
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = np.array(training_data['direction'])

X_test = np.array(test_data_[cols])
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = np.array(test_data['direction'])

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(lags, 1)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, verbose=False, validation_split=0.2, shuffle=False)
res = pd.DataFrame(model.history.history)
res[['accuracy', 'val_accuracy']].plot(figsize=(10, 6), style='--');
plt.figtext(0.5, 0.02, 'Fig. 1.3 Accuracy of the LSTM model on training and validation data per training step', style='italic',ha='center')
plt.show()

# Evaluate the model on the training data
train_loss, train_accuracy = model.evaluate(X_train, y_train)

# Training Data
# Make predictions on the training data
train_predictions = np.where(model.predict(X_train) > 0.5, 1, 0)

# Transforms the predictions into long-short positions, +1 and -1
training_data['prediction'] = np.where(train_predictions > 0, 1, -1)


# The number of the resulting short and long positions, respectively.
training_data['prediction'].value_counts()

# Calculates the strategy returns given the positions
training_data['strategy'] = training_data['prediction'] * training_data['returns']
training_data[['returns', 'strategy']].sum().apply(np.exp)
 
# Plots and compares the strategy performance to the benchmark performance (in-sample).
training_data[['returns', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6)); 
plt.figtext(0.5, 0.05, 'Fig. 1.4 Gross performance of EUR/USD compared to the DNNs-based strategy', style='italic',ha='center')
plt.figtext(0.5, -0.01, '(in-sample, no transaction costs)', style='italic',ha='center')

plt.show()

# Testing Data
# Evaluate the performance of the model on testing data
model.evaluate(X_test, y_test)

# Make predictions on the test data
test_predictions = np.where(model.predict(X_test) > 0.5, 1, 0)

# Transforms the predictions into long-short positions, +1 and -1
test_data['prediction'] = np.where(test_predictions > 0, 1, -1)

# The number of the resulting short and long positions, respectively.
test_data['prediction'].value_counts()

# Calculate the strategy returns given the positions
test_data['strategy'] = test_data['prediction'] * test_data['returns']
test_data[['returns', 'strategy']].sum().apply(np.exp)

# Plots and compares the strategy performance to the benchmark performance (out-of-sample)
test_data[['returns', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6)); 
plt.figtext(0.5, 0.05, 'Fig. 1.5 Gross performance of EUR/USD compared to the DNNs-based strategy', style='italic',ha='center')
plt.figtext(0.5, -0.01, '(out-of-sample, no transaction costs)', style='italic',ha='center')

plt.show()

### Adding Different Types of Features 
