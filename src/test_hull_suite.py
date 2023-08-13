import pandas as pd
# import talib as ta
import ta


def calculate_hma(data, window=9):
    close = data['close']
    hma = 2 * ta.sma(close, window=int(window / 2)) - ta.sma(close, window=window)
    hma = ta.sma(hma, window=int(window ** 0.5))
    return hma


def calculate_indicators(data):
    # Extract OHLCV data
    close = data['close']
    volume = data['volume']

    # Calculate HMA
    hma = calculate_hma(data, window=9)

    # Calculate HT
    ht = ta.trend.ema_indicator(close, window=9)

    # Calculate HO
    ho = ta.trend.ema_indicator(close, window=9) - ta.trend.ema_indicator(close, window=18)

    # Calculate RSI
    rsi = ta.momentum.rsi(close, window=14)

    # Calculate MACD
    macd = ta.trend.macd_diff(close, window_slow=26, window_fast=12, window_sign=9)
    signal = ta.trend.macd_signal(close, window_slow=26, window_fast=12, window_sign=9)

    return hma, ht, ho, rsi, macd, signal


def generate_signals(data):
    hma, ht, ho, rsi, macd, signal = calculate_indicators(data)

    buy_long_signals = (hma > ht) & (ht > ho) & (rsi < 30) & (macd > signal)
    sell_long_signals = (hma < ht) | (ht < ho) | (rsi > 70) | (macd < signal)

    buy_short_signals = (hma < ht) & (ht < ho) & (rsi > 70) & (macd < signal)
    sell_short_signals = (hma > ht) | (ht > ho) | (rsi < 30) | (macd > signal)

    signals = pd.DataFrame(index=data.index)
    signals['Buy Long'] = buy_long_signals
    signals['Sell Long'] = sell_long_signals
    signals['Buy Short'] = buy_short_signals
    signals['Sell Short'] = sell_short_signals

    return signals


# Example usage
data = pd.DataFrame({
    'open': [100, 105, 103, 108, 107],
    'high': [110, 112, 106, 115, 108],
    'low': [95, 100, 102, 104, 105],
    'close': [105, 110, 101, 109, 106],
    'volume': [1000, 2000, 1500, 1800, 1200]
})

signals = generate_signals(data)

print("Signals:")
print(signals)
