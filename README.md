# Stock-Market-Prediction

This repository contains a Streamlit web application for predicting stock prices using a Long Short-Term Memory (LSTM) model. The application fetches historical stock data, visualizes it, and predicts future prices based on past trends.

**Table of Contents**
Features
Installation
Usage
Model Training

**Features**
Fetches historical stock data from Yahoo Finance.
Visualizes stock closing prices, including 100-day and 200-day moving averages.
Splits data into training and testing sets.
Trains an LSTM model on the training data.
Predicts future stock prices based on the trained model.
Provides visual comparisons between predicted and actual stock prices.
Allows users to input the number of days for future price predictions.

**Installation**
Clone the repository.
Create and activate a virtual environment.
Install the required dependencies.
Download the pre-trained model and place it in the repository root directory, naming it keras_model.h5.

**Usage**
Run the Streamlit application.
Open your web browser and go to http://localhost:8501.
Enter a stock ticker symbol (e.g., AAPL) and explore the visualizations and predictions.

**Model Training**
If you need to train the LSTM model from scratch, ensure you have the necessary libraries installed. Then, run the script to train the model. The trained model will be saved as keras_model.h5.
