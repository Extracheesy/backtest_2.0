import sys
import pandas as pd
import pandas_ta as pta
sys.path.append('../..')

from utilities.backtesting import basic_single_asset_backtest, plot_wallet_vs_asset, get_metrics, get_n_columns, plot_sharpe_evolution, plot_bar_by_month
from utilities.custom_indicators import get_n_columns
from utilities.utils import get_n_columns

pd.options.mode.chained_assignment = None  # default='warn'


import matplotlib.pyplot as plt
import ta

"""
ds.features = {"close": None,
               "slope_is_dope": None,
               "n11_close": 11,
               "rsi": 7,
               "marketMA": 200,
               "fastMA": 21,
               "slowMA": 50,
               "entryMA": 3,
               "n11_entryMA": 11,
               "fast_slope": None,
               "slow_slope": None,
               "last_lowest": None
               }
"""

class SlopeIsDope():
    def __init__(
        self,
        df,
        type=["long"],
        shift_n11 = 11,
        window_rsi = 7,
        window_marketMA = 200,
        window_fastMA = 21,
        window_slowMA = 50,
        window_entryMA = 3,
        window_n11_entryMA = 11,
    ):
        self.df = df
        self.use_long = True if "long" in type else False
        self.use_short = True if "short" in type else False

        self.shift_n11 = shift_n11
        self.window_rsi = window_rsi
        self.window_marketMA = window_marketMA
        self.window_fastMA = window_fastMA
        self.window_slowMA = window_slowMA
        self.window_entryMA = window_entryMA
        self.window_n11_entryMA = window_n11_entryMA

    def populate_indicators(self):
        # -- Clear dataset --
        df = self.df
        df.drop(columns=df.columns.difference(['open','high','low','close','volume']), inplace=True)
        
        # -- Populate indicators --
        df['rsi'] = pta.rsi(close=df['close'], timeperiod=7)
        df['marketMA'] = pta.sma(close=df['close'], timeperiod=200)
        df['fastMA'] = pta.sma(close=df['close'], timeperiod=21)
        df['slowMA'] = pta.sma(close=df['close'], timeperiod=50)
        df['entryMA'] = pta.sma(close=df['close'], timeperiod=3)

        # Slow MA Y-axis A point
        df['sy1'] = df['slowMA'].shift(+11)
        df['sy2'] = df['slowMA'].shift(+1)
        sx1 = 1
        sx2 = 11
        df['sy'] = df['sy2'] - df['sy1']
        df['sx'] = sx2 - sx1
        df['slow_slope'] = df['sy'] / df['sx']
        df['fy1'] = df['fastMA'].shift(+11)
        df['fy2'] = df['fastMA'].shift(+1)
        fx1 = 1
        fx2 = 11
        df['fy'] = df['fy2'] - df['fy1']
        df['fx'] = fx2 - fx1
        df['fast_slope'] = df['fy'] / df['fx']

        df = get_n_columns(df, ["close", "entryMA"], 11)

        # ==== Trailing custom stoploss indicator ====
        df['last_lowest'] = df['low'].rolling(10).min().shift(1)
        df['slope_is_dope'] = True  # super_reversal indicator trigger

        lst_features = ["close", "entryMA"]

        df = get_n_columns(df, lst_features, self.shift_n11)

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
                (df['close'] > df['marketMA']) \
                & (df['fast_slope'] > 0) \
                & (df['slow_slope'] > 0) \
                # & (df['close'] > df['n11_entryMA']) \
                & (df['close'] > df['n11_close']) \
                & (df['rsi'] > 55) \
                # & (df['fastMA'] > df['slowMA'])
                , "open_long_market"
            ] = True
        
            # -- Populate close long market --
            df.loc[
                (df['fastMA'] < df['slowMA']) \
                | (df['close'] < df['last_lowest'])
                , "close_long_market"
            ] = True

        if self.use_short:
            # -- Populate open short market --
            """
            df.loc[
                (df['close_shift_5'] >= df['close']) \
                & (df['close_shift_10'] >= df['close_shift_5']) \
                & (df['close_shift_15'] >= df['close_shift_10'])
                , "open_short_market"
            ] = True
            """

            # -- Populate close short market --
            """
            df.loc[
                (df['close_shift_5'] <= df['close']) \
                & (df['close_shift_10'] <= df['close_shift_5']) \
                & (df['close_shift_15'] <= df['close_shift_10'])
                , "close_short_market"
            ] = True
            """
        
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
                    if row['close_long_market']:
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
                    if row['close_short_market']:
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
        


