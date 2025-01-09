import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_lstm_model(input_shape):
    """Build and compile the LSTM model."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_sequences(data, sequence_length=60):
    """Prepare training sequences for LSTM."""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Load and preprocess data (replace with your own path)
    from data_preprocessing import load_data, preprocess_data
    filepath = '../data/stock_prices.csv'
    data = load_data(filepath)
    scaled_data, _ = preprocess_data(data)

    # Prepare sequences and train the LSTM model
    sequence_length = 60
    X, y = prepare_sequences(scaled_data, sequence_length)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = create_lstm_model((X.shape[1], 1))
    model.fit(X, y, batch_size=32, epochs=10)
    model.save('../models/lstm_stock_model.h5')
    print("Model Trained and Saved!")
