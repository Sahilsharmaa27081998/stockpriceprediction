import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def predict_future(model, data, sequence_length=60):
    """Generate predictions for the future data."""
    last_sequence = data[-sequence_length:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    prediction = model.predict(last_sequence)
    return prediction

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data
    filepath = '../data/stock_prices.csv'
    data = load_data(filepath)
    scaled_data, scaler = preprocess_data(data)

    # Load trained model and make predictions
    model = load_model('../models/lstm_stock_model.h5')
    prediction = predict_future(model, scaled_data)

    # Inverse transform and visualize results
    prediction = scaler.inverse_transform(prediction)
    plt.plot(data.index[-10:], data['Close'].values[-10:], label='Actual')
    plt.plot(data.index[-1:], prediction[0], 'ro', label='Prediction')
    plt.legend()
    plt.show()
