import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """Load dataset from a CSV file."""
    data = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    return data

def preprocess_data(data, column='Close'):
    """Normalize and return the specified column for modeling."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[[column]])
    return data_scaled, scaler

if __name__ == "__main__":
    filepath = '../data/stock_prices.csv'
    data = load_data(filepath)
    scaled_data, scaler = preprocess_data(data)
    print("Data Preprocessed Successfully!")
