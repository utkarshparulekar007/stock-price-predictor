import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Download stock data
stock_symbol = "AAPL"
data = yf.download(stock_symbol, start="2018-01-01", end="2024-01-01")
data.dropna(inplace=True)

# Features and target
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']

# Train-test split (no shuffle for time-series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print("RMSE:", rmse)

print("R2 Score:", r2_score(y_test, y_pred))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual Price")
plt.plot(y_pred, label="Predicted Price")
plt.legend()
plt.title("Stock Price Prediction")
plt.savefig("output/prediction_plot.png")
plt.show()
