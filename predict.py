import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import date
import datetime
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler

st.write(
    """
    # Stock Price Analyzer

    Shown are the stock prices of Apple.
    """
)

ticker_symbol = st.text_input(
    "Enter the Symbol",
    "AAPL",
    key="placeholder",
)

col1, col2 = st.columns(2)

# Start date analysis
with col1:
    start_date = st.date_input("Input Starting Date",
                               datetime.date(2024, 1, 1))

# End date analysis
with col2:
    end_date = st.date_input("Input End Date",
                             value=date.today())

ticker_data = yf.Ticker(ticker_symbol)
ticker_df = ticker_data.history(period="1d",
                                start=f"{start_date}",
                                end=f"{end_date}")

st.write(f"""
{ticker_symbol}'s EOD Prices""")

st.dataframe(ticker_df)

st.write("""
        ## Daily Closing Price Chart
""")

# Plotly for interactive charts
fig = go.Figure()
fig.add_trace(go.Scatter(x=ticker_df.index, y=ticker_df['Close'],
                         mode='lines',
                         name='Close'))
fig.update_layout(title='Daily Closing Price',
                  xaxis_title='Date',
                  yaxis_title='Price (USD)',
                  hovermode='x unified')
st.plotly_chart(fig)

st.write("""
        ## Volume of Shares traded each day
""")

fig = go.Figure()
fig.add_trace(go.Scatter(x=ticker_df.index, y=ticker_df['Volume'],
                         mode='lines',
                         name='Volume'))
fig.update_layout(title='Volume of Shares Traded',
                  xaxis_title='Date',
                  yaxis_title='Volume',
                  hovermode='x unified')
st.plotly_chart(fig)

# Prepare the data for LSTM model
data = ticker_df.filter(['Close'])
dataset = data.values
training_data_len = int(np.ceil(len(dataset) * .95))

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data set
train_data = scaled_data[0:int(training_data_len), :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Plot with Plotly
st.write("""
        ## Model Predictions
""")
fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index, y=train['Close'],
                         mode='lines',
                         name='Train'))
fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'],
                         mode='lines',
                         name='Valid'))
fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'],
                         mode='lines',
                         name='Predictions'))
fig.update_layout(title='Model Predictions vs Actual Prices',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  hovermode='x unified')
st.plotly_chart(fig)

# Predict the next day's price
last_60_days = scaled_data[-60:]
last_60_days = np.reshape(last_60_days, (1, 60, 1))

next_day_prediction = model.predict(last_60_days)
next_day_prediction = scaler.inverse_transform(next_day_prediction)

st.write(f"""
        ## Next Day's Price Prediction
        The predicted closing price for the next trading day is: {next_day_prediction[0][0]:.2f}
""")
