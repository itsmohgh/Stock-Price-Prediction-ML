import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the historical stock data from the CSV file into a pandas DataFrame
stock_data = pd.read_csv('Iran Kh. Inv..csv', sep=',')

# Convert the date column to a datetime object for easier manipulation
stock_data['<DTYYYYMMDD>'] = pd.to_datetime(stock_data['<DTYYYYMMDD>'], format='%Y%m%d')

# Sort the DataFrame by date in ascending order
stock_data.sort_values(by='<DTYYYYMMDD>', ascending=True, inplace=True)

# Create a time series from the closing prices
time_series = stock_data.set_index('<DTYYYYMMDD>')['<CLOSE>']

# Check for and handle missing values if necessary
time_series = time_series.dropna()

# Prepare the data for the model
data = time_series.values.reshape(-1, 1)

# Scale the data to [0, 1] range
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Set the window size for sequences (you can adjust this as needed)
window_size = 12
# Create features and target variable for the model
X, y = [], []
for i in range(len(scaled_data) - window_size):
    X.append(scaled_data[i:i+window_size].flatten())
    y.append(scaled_data[i+window_size])

X, y = np.array(X), np.array(y)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Flatten y_train and y_test to 1-dimensional arrays
y_train = y_train.ravel()
y_test = y_test.ravel()

# Initialize the Gradient Boosting Regressor model
model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=3)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Inverse scale the predictions and actual values
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the original data and the predicted prices
plt.figure(figsize=(10, 6))
plt.plot(time_series.index[train_size+window_size:], y_test, label='Actual Prices', color='blue')
plt.plot(time_series.index[train_size+window_size:], y_pred, label='Predicted Prices', color='red')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction with Gradient Boosting Regressor')
plt.legend()
plt.show()

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")
