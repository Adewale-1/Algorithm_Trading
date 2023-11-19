import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import tensorflow as tf

#  Load datasets


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

    # Use the TensorFlow optimizer with gradient clipping
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


# Train LSTM model
def train_lstm_model(X_train, y_train, batch_size, epochs):
    X_train_reshaped = reshape_features_for_lstm(X_train)
    model = build_lstm_model((X_train_reshaped.shape[1], 1))

    # Early stopping to prevent overfitting
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

    # Plotting predictions vs actual values for visual inspection
    plt.plot(y_test, color='red', label='Actual EURUSD Price')
    plt.plot(y_pred, color='blue', label='Predicted EURUSD Price')
    plt.title('EURUSD Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('EURUSD Price')
    plt.legend()
    plt.show()

    return mse


if __name__ == "__main__":
    train_features_file = 'TrainedAndTestData/scaled_train_features.csv'
    train_targets_file = 'TrainedAndTestData/train_targets.csv'
    test_features_file = 'TrainedAndTestData/scaled_test_features.csv'
    test_targets_file = 'TrainedAndTestData/test_targets.csv'

    X_train, y_train, X_test, y_test = load_data(
        'TrainedAndTestData/scaled_train_features.csv',
        'TrainedAndTestData/train_targets.csv',
        'TrainedAndTestData/scaled_test_features.csv',
        'TrainedAndTestData/test_targets.csv'
    )

    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')

    model = train_lstm_model(X_train, y_train, batch_size=64, epochs=50)

    mse = evaluate_lstm_model(model, X_test, y_test)
