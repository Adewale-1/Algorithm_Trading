import pandas as pd

def load_data(price_file, prediction_file):
    prices = pd.read_csv(price_file)
    predictions = pd.read_csv(prediction_file)
    return prices, predictions

def backtest(prices, predictions):
    capital = 100  # Starting capital in USD
    position = 0  # No position initially
    trade_log = []  # Record of trades
    
    for i in range(1, len(prices)):
        if predictions[i] > prices[i-1] and position == 0:
            position = capital / prices[i]  # Buy at price[i]
            capital = 0
            trade_log.append(('BUY', prices[i], i))
        elif predictions[i] < prices[i-1] and position > 0:
            capital = position * prices[i]  # Sell at price[i]
            position = 0
            trade_log.append(('SELL', prices[i], i))
    
    # If a position is open at the end of the backtesting, close it
    if position > 0:
        capital = position * prices[-1]
    
    return capital, trade_log

if __name__ == "__main__":
    price_file = 'TrainedAndTestData/test_targets.csv'  # File with actual prices
    prediction_file = 'preprocessed_data.csv'  # File with predicted prices
    prices, predictions = load_data(price_file, prediction_file)
    
    final_capital, trade_log = backtest(prices['Close'].values, predictions['Predicted'].values)
    print(f"Final Capital: {final_capital}")
    for trade in trade_log:
        print(f"Trade: {trade}")