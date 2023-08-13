import sys

sys.path.append('../..')

from utilities.backtesting import basic_single_asset_backtest, plot_wallet_vs_asset, get_metrics, get_n_columns, plot_sharpe_evolution, plot_bar_by_month
from utilities.custom_indicators import get_n_columns

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

from datetime import datetime
from datetime import timedelta

import matplotlib.pyplot as plt
# import talib as ta
import ta

class ScalpingEngulfing():
    def __init__(
        self,
        df,
        type=["long"],
        bol_window = 200,
        bol_std = 2.25,
        min_bol_spread = 0,
        long_ma_window = 100,
        rsi_timeperiod=14,
    ):
        self.df = df
        self.use_long = True if "long" in type else False
        self.use_short = True if "short" in type else False
        self.bol_window = bol_window
        self.bol_std = bol_std
        self.min_bol_spread = min_bol_spread
        self.long_ma_window = long_ma_window
        self.rsi_timeperiod = rsi_timeperiod
        
    def populate_indicators(self):
        # -- Clear dataset --
        df = self.df
        df.drop(columns=df.columns.difference(['open','high','low','close','volume']), inplace=True)
        
        # -- Populate indicators --

        eng = talib.CDLENGULFING(df.open, df.high, df.low, df.close)

        df['eng'] = eng

        bol_band = ta.volatility.BollingerBands(close=df["close"], window=self.bol_window, window_dev=self.bol_std)
        df["lower_band"] = bol_band.bollinger_lband()
        df["higher_band"] = bol_band.bollinger_hband()
        df["ma_band"] = bol_band.bollinger_mavg()

        df['long_ma'] = ta.trend.sma_indicator(close=df['close'], window=self.long_ma_window)

        rsi = ta.momentum.RSIIndicator(close=df["close"], window=self.rsi_timeperiod)
        df["rsi"] = rsi.rsi()

        df['return'] = df['close'].pct_change()
        df['return'] = df['return'] * 100

        # df = get_n_columns(df, ["ma_band", "lower_band", "higher_band", "close"], 1)

        df.dropna(inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)

        self.df = df    
        return self.df
    
    def populate_buy_sell(self): 
        df = self.df
        # -- Initiate populate --
        df["open_long_market"] = False
        df["sum_return_open_long_market"] = 0
        df["close_long_market"] = False
        df["open_short_market"] = False
        df["sum_return_open_short_market"] = 0
        df["close_short_market"] = False
        
        if self.use_long:
            # -- Populate open long market --
            df.loc[
                ((df['close'] > df['long_ma'])
                & (df['rsi'] > 50)
                & (df['eng'] == 100))
                , "open_long_market"
            ] = True

            lst_buy_signal_idx = df[df["open_long_market"] == True].index.tolist()
            print('nb open_long', len(lst_buy_signal_idx))
            idx_time_end = df.index[-1]
            sl_tp = True
            cpt_sl_tp = 0
            tp = 0
            sl = 0
            if sl_tp:
                print('SL TP')
                for idx_time in lst_buy_signal_idx:
                    sum_return = 0
                    iter = 0
                    df["sum_return_open_long_market"] = 0
                    while sum_return < 3 and sum_return > -1:
                        idx_time_next = idx_time + timedelta(minutes=5)
                        iter += 1
                        if idx_time_next == idx_time_end:
                            df.at[idx_time_next, "close_long_market"] = True
                            print('BREAK')
                            break
                        try:
                            sum_return = sum_return + df.at[idx_time_next,"return"]
                            if sum_return >= 3 or sum_return <= -1:
                                if sum_return >= 3:
                                    tp += 1
                                else:
                                    sl += 1
                                print('SL TP long : ', cpt_sl_tp, ' = ', round(sum_return, 2), ' min: ', iter, ' sl: ', sl, ' tp: ', tp)
                                cpt_sl_tp += 1
                                df.at[idx_time_next, "close_long_market"] = True
                            idx_time = idx_time_next
                            df.at[idx_time,"sum_return_open_long_market"] = sum_return
                        except:
                            print('date not in index: ', idx_time_next)
                            idx_time = idx_time_next
                df["sum_return_open_long_market"] = 0
            else:
                # -- Populate close long market --
                df.loc[
                    ((df['close'] < df['long_ma'])
                    | (df['rsi'] < 30)
                    | (df['eng'] == -100))
                    , "close_long_market"
                ] = True

        if self.use_short:
            # -- Populate open short market --
            df.loc[
                ((df['close'] < df['long_ma'])
                & (df['rsi'] < 50)
                & (df['eng'] == -100))
                , "open_short_market"
            ] = True

            lst_buy_signal_idx = df[df["open_short_market"] == True].index.tolist()
            print('nb open_short', len(lst_buy_signal_idx))
            idx_time_end = df.index[-1]
            sl_tp = True
            cpt_sl_tp = 0
            tp = 0
            sl = 0
            if sl_tp:
                print('SL TP')
                for idx_time in lst_buy_signal_idx:
                    sum_return = 0
                    iter = 0
                    df["sum_return_open_short_market"] = 0
                    while sum_return < 1 and sum_return > -3:
                        idx_time_next = idx_time + timedelta(minutes=5)
                        iter += 1
                        if idx_time_next == idx_time_end:
                            df.at[idx_time_next, "close_short_market"] = True
                            print('BREAK')
                            break
                        try:
                            sum_return = sum_return + df.at[idx_time_next,"return"]
                            if sum_return >= 1 or sum_return <= -3:
                                if sum_return >= 1:
                                    sl += 1
                                else:
                                    tp += 1
                                print('SL TP short: ', cpt_sl_tp, ' = ', round(sum_return, 2), ' min: ', iter, ' sl: ', sl, ' tp: ', tp)
                                cpt_sl_tp += 1
                                df.at[idx_time_next, "close_short_market"] = True
                            idx_time = idx_time_next
                            df.at[idx_time,"sum_return_open_short_market"] = sum_return
                        except:
                            print('date not in index: ', idx_time_next)
                            idx_time = idx_time_next
                df["sum_return_open_short_market"] = 0
            else:
                # -- Populate close short market --
                df.loc[
                    (df['close'] > df['ma_band'])
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
        


