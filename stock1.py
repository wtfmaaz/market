import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.metrics import mean_squared_error
except ImportError as e:
    st.error(f"Missing library: {e}. Please install the required dependencies.")
    st.stop()

# Set Streamlit page configuration
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Helper function to create datasets
def create_dataset(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Main app
st.title("Stock Price Prediction using LSTM")
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Sidebar parameters
look_back = st.sidebar.slider("Look Back Days", 30, 120, 60)
epochs = st.sidebar.slider("Epochs", 10, 100, 20)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)

if uploaded_file is not None:
    try:
        # Load dataset
        data = pd.read_csv(uploaded_file)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        st.write("Dataset Preview:", data.head())

        # Train-test split
        train_data = data[:'2016']['Close'].values.reshape(-1, 1)
        test_data = data['2019':]['Close'].values.reshape(-1, 1)

        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(train_data)

        # Prepare train data
        X_train, y_train = create_dataset(scaled_data, look_back)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Prepare test data
        total_data = np.concatenate((train_data, test_data), axis=0)
        inputs = total_data[len(total_data) - len(test_data) - look_back:]
        inputs_scaled = scaler.transform(inputs)
        X_test, y_test = create_dataset(inputs_scaled, look_back)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        # Compile and train
        model.compile(optimizer='adam', loss='mean_squared_error')
        with st.spinner("Training the LSTM model..."):
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Predictions
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        # Plot results
        st.subheader("Actual vs Predicted Prices")
        plt.figure(figsize=(12, 6))
        plt.plot(test_data, color='blue', label='Actual Prices')
        plt.plot(predictions, color='red', label='Predicted Prices')
        plt.title("Stock Price Prediction")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(plt)

        # Future predictions
        future_predictions = []
        input_seq = inputs_scaled[-look_back:]
        for _ in range(30):  # Predict next 30 days
            pred_scaled = model.predict(input_seq.reshape(1, look_back, 1))[0, 0]
            pred_unscaled = scaler.inverse_transform([[pred_scaled]])[0, 0]
            future_predictions.append(pred_unscaled)
            input_seq = np.append(input_seq[1:], pred_scaled).reshape(look_back, 1)

        st.subheader("Next 30 Days Prediction")
        st.line_chart(future_predictions)

        # Metrics
        rmse = np.sqrt(mean_squared_error(test_data[:len(predictions)], predictions))
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please upload a dataset to proceed.")
