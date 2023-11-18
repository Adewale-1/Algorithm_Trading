import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import talib

# Load dataset
def load_data(filename):
    df = pd.read_csv(filename, index_col='Date', parse_dates=True)
    return df

# Clean data
def clean_data(df):
    # Drop any rows with missing values
    df = df.dropna()
    return df

# Add technical indicators as features
def add_technical_indicators(df):
    # Add moving averages
    df['SMA_50'] = talib.SMA(df['Close'].values, timeperiod=50)
    df['SMA_200'] = talib.SMA(df['Close'].values, timeperiod=200)

    # Add RSI
    df['RSI_14'] = talib.RSI(df['Close'].values, timeperiod=14)

    # Add MACD
    macd, macdsignal, _ = talib.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_signal'] = macdsignal

    # Add Bollinger Bands
    upperband, middleband, lowerband = talib.BBANDS(df['Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_upper'] = upperband
    df['BB_middle'] = middleband
    df['BB_lower'] = lowerband

    return df

# Normalize data
def normalize_data(df):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df_scaled

# Main pre-processing function
def preprocess_data(filename):
    df = load_data(filename)
    df_cleaned = clean_data(df)
    df_features = add_technical_indicators(df_cleaned)
    df_final = normalize_data(df_features)
    return df_final

# Split data into training and testing sets
def split_data(df, test_size=0.2):
    feature_columns = df.columns.difference(['Close'])
    X = df[feature_columns]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    filename = 'EURUSD=X >>> 1d >>> 2020-01-01 >>> 2023-11-17.csv'  # Mutable with the filename
    df_preprocessed = preprocess_data(filename)
    X_train, X_test, y_train, y_test = split_data(df_preprocessed)
    
    # Optionally, save the preprocessed data to new CSV files
    df_preprocessed.to_csv('preprocessed_data.csv')
    X_train.to_csv('train_features.csv')
    y_train.to_csv('train_targets.csv')
    X_test.to_csv('test_features.csv')
    y_test.to_csv('test_targets.csv')