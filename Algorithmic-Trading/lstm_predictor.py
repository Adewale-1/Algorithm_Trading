import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import os

# Load datasets


def load_data(train_features_file, train_targets_file, test_features_file, test_targets_file):
    X_train = pd.read_csv(train_features_file)
    y_train = pd.read_csv(train_targets_file)
    X_test = pd.read_csv(test_features_file)
    y_test = pd.read_csv(test_targets_file)
    return X_train, y_train.values.ravel(), X_test, y_test.values.ravel()

# Reshape features for LSTM


def reshape_features_for_lstm(X):
    return np.reshape(X.values, (X.shape[0], X.shape[1], 1))

# Build LSTM model


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Train LSTM model


def train_lstm_model(X_train, y_train, batch_size, epochs):
    X_train_reshaped = reshape_features_for_lstm(X_train)
    model = build_lstm_model((X_train_reshaped.shape[1], 1))
    early_stopping = EarlyStopping(
        monitor='val_loss', mode='min', verbose=1, patience=10)
    model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size,
              verbose=1, validation_split=0.2, callbacks=[early_stopping])
    return model

# Evaluate LSTM model


def evaluate_lstm_model(model, X_test, y_test):
    X_test_reshaped = reshape_features_for_lstm(X_test)
    y_pred = model.predict(X_test_reshaped)
    mse = mean_squared_error(y_test, y_pred)
    print(f"The Mean Squared Error on the test set is: {mse}")
    return mse


# Main script execution
if __name__ == "__main__":
    current_directory = os.getcwd()
    print(f"The current working directory is: {current_directory}")

    train_features_file = 'TrainedAndTestData/scaled_train_features.csv'
    train_targets_file = 'TrainedAndTestData/train_targets.csv'
    test_features_file = 'TrainedAndTestData/scaled_test_features.csv'
    test_targets_file = 'TrainedAndTestData/test_targets.csv'

    X_train, y_train, X_test, y_test = load_data(
        train_features_file, train_targets_file, test_features_file, test_targets_file)

    model = train_lstm_model(X_train, y_train, batch_size=64, epochs=50)
    mse = evaluate_lstm_model(model, X_test, y_test)

    try:
        print("Generating predictions...\n")
        X_test_reshaped = reshape_features_for_lstm(X_test)
        predicted_prices = model.predict(X_test_reshaped)
        predicted_prices = predicted_prices.flatten()

        predictions_filename = 'Predictions/model_predictions.csv'
        predictions_df = pd.DataFrame(predicted_prices, columns=['Predicted'])
        predictions_df.to_csv(predictions_filename, index=False)

        print(f"Model predictions saved to {predictions_filename}")
    except Exception as e:
        print(f"Failed to generate or save predictions: {e}")
