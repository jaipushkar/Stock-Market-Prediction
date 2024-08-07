import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

start = '2013-01-01'
end = '2023-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)

# Check if data is successfully fetched
if df.empty:
    st.error("No data fetched. Please check the stock ticker or the date range.")
else:
    # Describing Data
    st.subheader('Data from 2013 to 2023')
    st.write(df.describe())

    # Visualizations
    st.subheader('Closing Price vs Time Chart')
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df.Close)
    st.pyplot(fig1)

    st.subheader('Closing Price vs Time Chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(ma100, label='100MA')
    ax2.plot(df.Close, label='Closing Price')
    ax2.legend()
    st.pyplot(fig2)

    st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(ma100, label='100MA')
    ax3.plot(ma200, label='200MA')
    ax3.plot(df.Close, label='Closing Price')
    ax3.legend()
    st.pyplot(fig3)

    # Splitting data into training and testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

    # Check if training data is not empty
    if data_training.empty or data_testing.empty:
        st.error("Training or testing data is empty. Please check the stock ticker or the date range.")
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        # Load the model
        model = load_model('keras_model.h5')

        # Prepare test data
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df)

        X_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            X_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

        X_test, y_test = np.array(X_test), np.array(y_test)
        y_predicted = model.predict(X_test)

        scaler_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scaler_factor
        y_test = y_test * scaler_factor

        # Display predicted prices
        st.subheader('Predicted Prices')
        predicted_prices_df = pd.DataFrame({
            'Original Price': y_test,
            'Predicted Price': y_predicted.flatten()
        })
        st.write(predicted_prices_df)

        # Final Graph
        st.subheader('Prediction vs Original')
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        ax4.plot(y_test, 'b', label='Original Price')
        ax4.plot(y_predicted, 'r', label='Predicted Price')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Price')
        ax4.legend()
        st.pyplot(fig4)

        # Predict future prices
        st.subheader('Future Price Prediction')
        future_days = st.number_input('Enter number of days to predict', min_value=1, value=30)
        future_input = input_data[-100:]
        future_predictions = []

        for _ in range(future_days):
            future_pred = model.predict(future_input.reshape(1, 100, 1))
            future_predictions.append(future_pred[0, 0])
            future_input = np.append(future_input[1:], future_pred, axis=0)

        future_predictions = np.array(future_predictions) * scaler_factor

        future_dates = pd.date_range(start=df.index[-1], periods=future_days + 1, inclusive='right')
        future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Future Price'])

        st.write(future_df)

        fig5, ax5 = plt.subplots(figsize=(12, 6))
        ax5.plot(df.Close, label='Historical Prices')
        ax5.plot(future_df, label='Future Prices', color='r')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Price')
        ax5.legend()
        st.pyplot(fig5)
