import sys

sys.path.append('../..')

from utilities.backtesting import basic_single_asset_backtest, plot_wallet_vs_asset, get_metrics, get_n_columns, plot_sharpe_evolution, plot_bar_by_month
from utilities.custom_indicators import get_n_columns

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


import matplotlib.pyplot as plt
import ta
from ta.trend import macd
from ta.momentum import rsi


class HullSuite():
    def __init__(
        self,
        df,
        type=["long"],
        short_window = 9,
        long_window = 18,
        SL = 0,
        TP = 0
    ):
        self.df = df
        self.use_long = True if "long" in type else False
        self.use_short = True if "short" in type else False
        self.short_window = short_window
        self.long_window = long_window

        self.SL = SL
        self.TP = TP
        if self.SL == 0:
            self.SL = -10000
        if self.TP == 0:
            self.TP = 10000

    def calculate_hma(self, data, window=9):
        close = data['close']
        hma = 2 * ta.trend.sma_indicator(close=close, window=int(window / 2)) - ta.trend.sma_indicator(close=close, window=window)
        hma = ta.trend.ema_indicator(close=hma, window=int(window ** 0.5))
        return hma

    def populate_indicators(self):
        # -- Clear dataset --
        df = self.df
        df.drop(columns=df.columns.difference(['open','high','low','close','volume']), inplace=True)

        # -- Populate indicators --

        # Calculate HMA
        # df["hma"] = ta.trend.hull_moving_average(close=df["close"], window=9)
        df["hma"] = self.calculate_hma(df, self.short_window)

        # Calculate HT
        df["ht"] = ta.trend.ema_indicator(close=df["close"], window=self.short_window)

        # Calculate HO
        df["ho"] = ta.trend.ema_indicator(close=df["close"], window=self.short_window) - ta.trend.ema_indicator(close=df["close"], window=18)

        # Calculate RSI
        df["rsi"] = ta.momentum.rsi(close=df['close'], window=14, fillna=True)

        # Calculate MACD
        df["macd"] = ta.trend.macd_diff(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df["signal"] = ta.trend.macd_signal(close=df['close'], window_slow=26, window_fast=12, window_sign=9)

        self.df = df
        return self.df
    
    def populate_buy_sell(self): 
        df = self.df
        # -- Initiate populate --
        df["open_long_market"] = False
        df["close_long_market"] = False
        df["open_short_market"] = False
        df["close_short_market"] = False
        
        if self.use_long:
            # -- Populate open long market --
            df.loc[
                (df['hma'] > df['ht'])
                & (df['ht'] > df['ho'])
                # & (df['rsi'] < 30)
                & (df['macd'] > df['signal'])
                , "open_long_market"
            ] = True
        
            # -- Populate close long market --
            df.loc[
                (df['hma'] < df['ht'])
                | (df['ht'] < df['ho'])
                | (df['rsi'] > 70)
                | (df['macd'] < df['signal'])
                , "close_long_market"
            ] = True

        if self.use_short:
            # -- Populate open short market --
            df.loc[
                (df['hma'] < df['ht'])
                & (df['ht'] < df['ho'])
                # & (df['rsi'] > 70)
                & (df['macd'] < df['signal'])
                , "open_short_market"
            ] = True
        
            # -- Populate close short market --
            df.loc[
                (df['hma'] > df['ht'])
                | (df['ht'] > df['ho'])
                | (df['rsi'] < 30)
                | (df['macd'] > df['signal'])
                , "close_short_market"
            ] = True
        
        self.df = df   
        return self.df
        
    def run_backtest(self, initial_wallet=1000, leverage=1):
        df = self.df[:]
        wallet = initial_wallet
        maker_fee = 0.0002
        taker_fee = 0.0007
        trades = []
        days = []
        current_day = 0
        previous_day = 0
        current_position = None

        for index, row in df.iterrows():
            
            # -- Add daily report --
            current_day = index.day
            if previous_day != current_day:
                temp_wallet = wallet
                if current_position:
                    if current_position['side'] == "LONG":
                        close_price = row['close']
                        trade_result = (close_price - current_position['price']) / current_position['price']
                        temp_wallet += temp_wallet * trade_result
                        fee = temp_wallet * taker_fee
                        temp_wallet -= fee
                    elif current_position['side'] == "SHORT":
                        close_price = row['close']
                        trade_result = (current_position['price'] - close_price) / current_position['price']
                        temp_wallet += temp_wallet * trade_result
                        fee = temp_wallet * taker_fee
                        temp_wallet -= fee
                    
                days.append({
                    "day":str(index.year)+"-"+str(index.month)+"-"+str(index.day),
                    "wallet":temp_wallet,
                    "price":row['close']
                })
            previous_day = current_day
            if current_position:
            # -- Check for closing position --
                if current_position['side'] == "LONG":
                    # -- Close LONG market --
                    if row['close_long_market'] or self.action_sl_tp(row, current_position, leverage, wallet):  # MODIF CEDE ADD SL AND TF IN THIS TEST
                        close_price = row['close']
                        trade_result = ((close_price - current_position['price']) / current_position['price']) * leverage
                        wallet += wallet * trade_result
                        fee = wallet * taker_fee
                        wallet -= fee
                        trades.append({
                            "open_date": current_position['date'],
                            "close_date": index,
                            "position": "LONG",
                            "open_reason": current_position['reason'],
                            "close_reason": "Market",
                            "open_price": current_position['price'],
                            "close_price": close_price,
                            "open_fee": current_position['fee'],
                            "close_fee": fee,
                            "open_trade_size":current_position['size'],
                            "close_trade_size": wallet,
                            "wallet": wallet
                        })
                        current_position = None
                        
                elif current_position['side'] == "SHORT":
                    # -- Close SHORT Market --
                    if row['close_short_market'] or self.action_sl_tp(row, current_position, leverage, wallet): # MODIF CEDE ADD SL AND TF IN THIS TEST
                        close_price = row['close']
                        trade_result = ((current_position['price'] - close_price) / current_position['price']) * leverage
                        wallet += wallet * trade_result
                        fee = wallet * taker_fee
                        wallet -= fee
                        trades.append({
                            "open_date": current_position['date'],
                            "close_date": index,
                            "position": "SHORT",
                            "open_reason": current_position['reason'],
                            "close_reason": "Market",
                            "open_price": current_position['price'],
                            "close_price": close_price,
                            "open_fee": current_position['fee'],
                            "close_fee": fee,
                            "open_trade_size": current_position['size'],
                            "close_trade_size": wallet,
                            "wallet": wallet
                        })
                        current_position = None

            # -- Check for opening position --
            else:
                # -- Open long Market --
                if row['open_long_market']:
                    open_price = row['close']
                    fee = wallet * taker_fee
                    wallet -= fee
                    pos_size = wallet
                    current_position = {
                        "size": pos_size,
                        "date": index,
                        "price": open_price,
                        "fee":fee,
                        "reason": "Market",
                        "side": "LONG",
                    }
                elif row['open_short_market']:
                    open_price = row['close']
                    fee = wallet * taker_fee
                    wallet -= fee
                    pos_size = wallet
                    current_position = {
                        "size": pos_size,
                        "date": index,
                        "price": open_price,
                        "fee":fee,
                        "reason": "Market",
                        "side": "SHORT"
                    }
                    
                    
        df_days = pd.DataFrame(days)
        df_days['day'] = pd.to_datetime(df_days['day'])
        df_days = df_days.set_index(df_days['day'])

        df_trades = pd.DataFrame(trades)
        if df_trades.empty:
            print("!!! No trades")
            return None
        else:
            df_trades['open_date'] = pd.to_datetime(df_trades['open_date'])
            df_trades = df_trades.set_index(df_trades['open_date'])  
        
        return get_metrics(df_trades, df_days) | {
            "wallet": wallet,
            "trades": df_trades,
            "days": df_days
        }

    def action_sl_tp(self, row, position, leverage, wallet):
        close_price = row['close']
        if position["side"] == "LONG":
            trade_result = ((close_price - position['price']) / position['price']) * leverage * 100
            if trade_result <= self.SL or trade_result >= self.TP:
                print("SL TP")
                return True
            else:
                return False
        elif position["side"] == "SHORT":
            trade_result = ((position['price'] - close_price) / position['price']) * leverage * 100
            if trade_result <= self.SL or trade_result >= self.TP:
                print("SL TP")
                return True
            else:
                return False
        return False
        


