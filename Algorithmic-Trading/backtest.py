import pandas as pd

# Load datasets


def load_data(price_file, prediction_file):
    prices = pd.read_csv(price_file)
    predictions = pd.read_csv(prediction_file)
    return prices, predictions

# Backtest strategy implementation
def backtest(prices, predictions):
    capital = 100  # Starting capital
    position = 0  # Indicates no position
    trade_log = []  # Logs all the trades

    for i in range(1, len(prices)):
        if predictions[i] > prices[i-1] and position == 0:  # Buy signal
            position = capital / prices[i]   # Buy
            capital = 0
            trade_log.append(('BUY', prices[i], i))
        elif predictions[i] < prices[i-1] and position > 0:  # Sell signal
            capital = position * prices[i]  # Sell
            position = 0
            trade_log.append(('SELL', prices[i], i))

    # Closing any open positions on the last day
    if position > 0:
        capital = position * prices[-1]  # Sell

    return capital, trade_log


if __name__ == "__main__":
    price_file = 'TrainedAndTestData/test_targets.csv'          # Actual prices
    prediction_file = 'Predictions/model_predictions.csv'  # Model's predictions

    prices, predictions = load_data(price_file, prediction_file)

    final_capital, trade_log = backtest(
        prices['Close'].values, predictions['Predicted'].values)
    print(f"Final Capital: {final_capital}")
    for trade in trade_log:
        print(f"Trade: {trade}")
