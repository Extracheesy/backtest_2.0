import sys

sys.path.append('../..')

from utilities.backtesting import basic_single_asset_backtest, plot_wallet_vs_asset, get_metrics, get_n_columns, plot_sharpe_evolution, plot_bar_by_month
from utilities.custom_indicators import get_n_columns

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


import matplotlib.pyplot as plt
import ta
from ta.trend import macd


class BollingerReversion():
    def __init__(
        self,
        df,
        type=["long"],

        # -- Indicator variable --
        bol_window=100,
        bol_std=2.25,
        min_bol_spread=0,
        long_ma_window=500,

        stochOverBought = 0.8,
        stochOverSold = 0.2,
        willOverSold = -80,
        willOverBought = -25,

        SL = 0,
        TP = 0
    ):
        self.df = df
        self.use_long = True if "long" in type else False
        self.use_short = True if "short" in type else False

        self.aoParam1 = 6
        self.aoParam2 = 22
        self.stochWindow = 14
        self.willWindow = 14

        # -- Hyper parameters --
        self.bol_window = bol_window
        self.bol_std = bol_std

        self.stochOverBought = stochOverBought
        self.stochOverSold = stochOverSold
        self.willOverSold = willOverSold
        self.willOverBought = willOverBought
        self.long_ma_window = long_ma_window
        self.TpPct = 0.15
        self.SL = SL
        self.TP = TP
        if self.SL == 0:
            self.SL = -10000
        if self.TP == 0:
            self.TP = 10000
        
    def populate_indicators(self):
        # -- Clear dataset --
        df = self.df
        df.drop(columns=df.columns.difference(['open','high','low','close','volume']), inplace=True)
        
        # -- Populate indicators --

        # -- Indicators, you can edit every value --
        df['AO'] = ta.momentum.awesome_oscillator(df['high'], df['low'], window1=self.aoParam1, window2=self.aoParam2)
        df['previousRow_AO'] = df['AO'].shift(1)
        df['STOCH_RSI'] = ta.momentum.stochrsi(close=df['close'], window=self.stochWindow)
        df['WillR'] = ta.momentum.williams_r(high=df['high'], low=df['low'], close=df['close'], lbp=self.willWindow)
        df['EMA100'] = ta.trend.ema_indicator(close=df['close'], window=100)
        df['EMA200'] = ta.trend.ema_indicator(close=df['close'], window=200)

        bol_band = ta.volatility.BollingerBands(close=df["close"], window=self.bol_window, window_dev=self.bol_std)
        df["lower_band"] = bol_band.bollinger_lband()
        df["higher_band"] = bol_band.bollinger_hband()
        df["ma_band"] = bol_band.bollinger_mavg()

        df['long_ma'] = ta.trend.sma_indicator(close=df['close'], window=self.long_ma_window)

        df = get_n_columns(df, ["ma_band", "lower_band", "higher_band", "close"], 1)

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
                (df['AO'] > 0)
                & ((df['close'] < df['lower_band'])
                   | (df['close'] > df['higher_band']))
                & ((df['STOCH_RSI'] < self.stochOverSold)
                   | (df['WillR'] < self.willOverSold))
                 , "open_long_market"
            ] = True
        
            # -- Populate close long market --
            df.loc[
                (df['close'] >= df['ma_band'])
                & (df['n1_close'] <= df['n1_ma_band'])
                | (df['close'] <= df['ma_band'])
                & (df['n1_close'] >= df['n1_ma_band'])
                , "close_long_market"
            ] = True

        if self.use_short:
            # -- Populate open short market --
            df.loc[
                (df['AO'] < 0)
                & ((df['close'] < df['lower_band'])
                   | (df['close'] > df['higher_band']))
                & ((df['STOCH_RSI'] > self.stochOverBought)
                   | (df['WillR'] > self.willOverBought))
                , "open_short_market"
            ] = True
        
            # -- Populate close short market --
            df.loc[
                (df['close'] >= df['ma_band'])
                & (df['n1_close'] <= df['n1_ma_band'])
                | (df['close'] <= df['ma_band'])
                & (df['n1_close'] >= df['n1_ma_band'])
                , "close_short_market"
            ] = True
        
        self.df = df
        self.df_engaged = pd.DataFrame(index=self.df.index)
        return self.df

    def fill_df_open_close(self, df, pair):
        status = "FREE"
        for idx in df.index:
            if df.at[idx, pair] == 'CLOSE':
                df.at[idx, pair] = True
                status = 'FREE'
            elif df.at[idx, pair] == 'OPEN' \
                    or status == "ENGAGED":
                df.at[idx, pair] = True
                status = "ENGAGED"
        return df


    def get_df_engaged(self, pair, bt_result_in):
        bt_result = bt_result_in.copy()
        self.df_engaged[pair] = False
        lst_open = bt_result['trades']['open_date'].to_list()
        lst_close = bt_result['trades']['close_date'].to_list()

        lst_open_val = ["OPEN"] * len(lst_open)
        self.df_engaged.loc[lst_open, pair] = lst_open_val

        lst_close_val = ["CLOSE"] * len(lst_open)
        self.df_engaged.loc[lst_close, pair] = lst_close_val
        self.df_engaged = self.fill_df_open_close(self.df_engaged, pair)

        return self.df_engaged


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
        


