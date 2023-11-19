import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import talib
from datetime import datetime

# Load dataset


def load_data(filename):
    df = pd.read_csv(filename, index_col='Datetime', parse_dates=True)
    return df

# Clean data


def clean_data(df):
    # Remove NaNs which might affect indicators calculation
    df_cleaned = df.dropna()
    return df_cleaned

# Add technical indicators as features


def add_technical_indicators(df):
    # Add Simple Moving Averages
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)

    # Add Relative Strength Index
    df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)

    # Add Moving Average Convergence Divergence
    macd, macdsignal, _ = talib.MACD(
        df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal

    # Add Bollinger Bands
    upperband, middleband, lowerband = talib.BBANDS(
        df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['Upper_BB'] = upperband
    df['Middle_BB'] = middleband
    df['Lower_BB'] = lowerband

    # Additional indicators
    df['ATR_14'] = talib.ATR(
        df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
    df['CCI_14'] = talib.CCI(
        df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
    df['STOCH_k'], df['STOCH_d'] = talib.STOCH(df['High'].values, df['Low'].values, df['Close'].values,
                                               fastk_period=14, slowk_period=3, slowk_matype=0,
                                               slowd_period=3, slowd_matype=0)

    # Drop rows with NaN values introduced by indicator windows
    df = df.dropna()
    return df

# Scale features with a StandardScaler


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

# Main preprocessing function


def preprocess_data(filename):
    df = load_data(filename)
    df_cleaned = clean_data(df)
    df_indicators = add_technical_indicators(df_cleaned)

    # Extract the Close price to be the target variable
    target = df_indicators.pop('Close')
    features = df_indicators

    # Normalize the feature data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, shuffle=False)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test


if __name__ == "__main__":
    end_date = datetime.now().strftime('%Y-%m-%d')
    # replace with your CSV file
    filename = f"CurrencyData/EURUSD=X >>> 1h >>> 2022-01-01 >>> {end_date}.csv"

    # Run the preprocessing
    X_train_scaled, y_train, X_test_scaled, y_test = preprocess_data(filename)

    # Save the preprocessed data to CSV files for later use
    pd.DataFrame(X_train_scaled).to_csv(
        'TrainedAndTestData/scaled_train_features.csv', index=False)
    y_train.to_csv('TrainedAndTestData/train_targets.csv', index=False)
    pd.DataFrame(X_test_scaled).to_csv(
        'TrainedAndTestData/scaled_test_features.csv', index=False)
    y_test.to_csv('TrainedAndTestData/test_targets.csv', index=False)