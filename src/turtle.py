import sys

sys.path.append('../..')

from utilities.backtesting import basic_single_asset_backtest, plot_wallet_vs_asset, get_metrics, get_n_columns, plot_sharpe_evolution, plot_bar_by_month
from utilities.custom_indicators import get_n_columns

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


import matplotlib.pyplot as plt
# import talib as ta
import ta

class Turtle():
    def __init__(
        self,
        df,
        type=["long"],
        SL = 0,
        TP = 0
    ):
        self.df = df
        self.use_long = True if "long" in type else False
        self.use_short = True if "short" in type else False
        self.SL = SL
        self.TP = TP
        if self.SL == 0:
            self.SL = -10000
        if self.TP == 0:
            self.TP = 10000
        
    def populate_indicators(self):
        # -- Clear dataset --
        data = self.df
        data.drop(columns=data.columns.difference(['open','high','low','close','volume']), inplace=True)

        # -- Populate indicators --
        # Calculate SMA20 and SMA55
        data['SMA20'] = ta.trend.sma_indicator(data['close'], window=20)
        data['SMA55'] = ta.trend.sma_indicator(data['close'], window=55)

        # Calculate Donchian Channel
        data['UpperChannel'] = data['high'].rolling(window=20).max()
        data['LowerChannel'] = data['low'].rolling(window=20).min()

        # Calculate stop loss and take profit levels
        data['StopLossLong'] = data['LowerChannel'].rolling(window=2).min().shift()
        data['StopLossShort'] = data['UpperChannel'].rolling(window=2).max().shift()
        data['TakeProfitLong'] = data['UpperChannel'].rolling(window=10).max().shift()
        data['TakeProfitShort'] = data['LowerChannel'].rolling(window=10).min().shift()

        self.df = data
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
                # Generate entry signals for long positions
                (df['close'] > df['UpperChannel'].shift(1))
                & (df['SMA20'] > df['SMA55'])
                , "open_long_market"
            ] = True
        
            # -- Populate close long market --
            df.loc[
                (df['close'] < df['LowerChannel'].rolling(window=10).min())
                | (df['close'] <= df['StopLossLong'])
                , "close_long_market"
            ] = True

        if self.use_short:
            # -- Populate open short market --
            df.loc[
                (df['close'] < df['LowerChannel'].shift(1))
                & (df['SMA20'] < df['SMA55'])
                , "open_short_market"
            ] = True
        
            # -- Populate close short market --
            df.loc[
                (df['close'] > df['UpperChannel'].rolling(window=10).max())
                | (df['close'] >= df['StopLossShort'])
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
        


