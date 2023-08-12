# Reference:
# https://www.dutchalgotrading.com/strategies/the-clucmay72018-algo-trading-strategy/

import sys

sys.path.append('../..')

from utilities.backtesting import basic_single_asset_backtest, plot_wallet_vs_asset, get_metrics, get_n_columns, plot_sharpe_evolution, plot_bar_by_month, plot_df_data
from utilities.custom_indicators import get_n_columns

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


import matplotlib.pyplot as plt
import talib as ta
# from ta.momentum import RSIIndicator
# from ta.trend import macd

class ClucMay():
    def __init__(
        self,
        df,
        type=["long"],
        # bol_window = 100,
        bol_window=20,
        # bol_std = 2.25,
        bol_std=2,
        min_bol_spread = 0,
        long_ma_window = 500,
        rsi_timeperiod = 2,
        ema_rsi_timeperiod = 5,
        ema_timeperiod = 50,
        SL=0,
        TP=0
    ):
        self.df = df
        self.use_long = True if "long" in type else False
        self.use_short = True if "short" in type else False
        self.bol_window = bol_window
        self.bol_std = bol_std
        self.min_bol_spread = min_bol_spread
        self.long_ma_window = long_ma_window
        self.rsi_timeperiod = rsi_timeperiod
        self.ema_rsi_timeperiod = ema_rsi_timeperiod
        self.ema_timeperiod = ema_timeperiod

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

        """
        # -- Populate indicators --
        bol_band = ta.volatility.BollingerBands(close=df["close"], window=self.bol_window, window_dev=self.bol_std)
        df["lower_band"] = bol_band.bollinger_lband()
        df["higher_band"] = bol_band.bollinger_hband()
        df["ma_band"] = bol_band.bollinger_mavg()

        df['long_ma'] = ta.trend.sma_indicator(close=df['close'], window=self.long_ma_window)

        df = get_n_columns(df, ["ma_band", "lower_band", "higher_band", "close"], 1)
        
        self.df = df    
        return self.df
        """

        # -- Populate indicators --
        rsi = ta.momentum.RSIIndicator(close=df["close"], window=self.rsi_timeperiod)
        df["rsi"] = rsi.rsi()
        rsiframe = pd.DataFrame(df['rsi']).rename(columns={'rsi': 'close'})
        df['emarsi'] = ta.trend.sma_indicator(close=rsiframe['close'], window=self.ema_rsi_timeperiod)

        df["macd"] = macd(df['close'], window_slow=26, window_fast=12)

        df_adx = df.ta.adx()
        df['adx'] = df_adx['ADX_14']

        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        bollinger = ta.volatility.BollingerBands(close=df["close"], window=self.bol_window, window_dev=self.bol_std)
        # df['bb_lowerband'] = bollinger['lower']
        df['bb_lowerband'] = bollinger.bollinger_lband()

        # df['bb_middleband'] = bollinger['mid']
        df["bb_middleband"] = bollinger.bollinger_mavg()
        # df['bb_upperband'] = bollinger['upper']
        df["bb_upperband"] = bollinger.bollinger_hband()
        # df['ema100'] = ta.EMA(df, timeperiod=self.ema_timeperiod)
        df['ema100'] = ta.trend.ema_indicator(close=df['close'], window=self.ema_timeperiod)

        df['bb_lowerband_0985'] = 0.985 * df['bb_lowerband']
        df['volume_rolling'] = df['volume'].rolling(window=30).mean().shift(1) * 20

        df['bb_lowerband_middle'] = df['bb_lowerband'] - (df['close'] - df['bb_lowerband']) / 2
        # df['bb_upperband_middle'] = df['bb_upperband'] + (df['close'] - df['bb_upperband']) / 2
        df['bb_upperband_middle'] = df['bb_upperband'] - (df['bb_upperband'] - df['close']) / 2
        # df['bb_upperband_middle_3'] = df['bb_upperband_middle_1'] - df['bb_upperband_middle_2']
        # plot_df_data(df)

        df = df.dropna()

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
                (
                # (df['close'] < df['ema100']) &
                # (df['close'] < 0.985 * df['bb_lowerband'])
                # (df['close'] <  df['bb_lowerband'] - df['bb_lowerband'] * (1 - 0.985))
                (df['close'] < df['bb_lowerband_middle'])
                & (df['volume'] < (df['volume'].rolling(window=30).mean().shift(1) * 20))
                & (df["macd"] > 0)
                )
                , "open_long_market"
            ] = True
        
            # -- Populate close long market --
            df.loc[
                (df['close'] > df['bb_middleband'])
                , "close_long_market"
            ] = True

        if self.use_short:
            # -- Populate open short market --
            df.loc[
                (
                # (df['close'] > df['ema100']) &
                # (df['close'] > 0.985 * df['bb_upperband'])
                (df['close'] > df['bb_upperband_middle'])
                # & (df['volume'] < (df['volume'].rolling(window=30).mean().shift(1) * 20))
                & (df["macd"] < 0)
                )
                , "open_short_market"
            ] = True
        
            # -- Populate close short market --
            df.loc[
                (df['close'] < df['bb_middleband'])
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

                    # if str(current_position['date']) == "2023-01-24 21:50:00":
                    # if str(current_position['date']) == "2022-01-05 18:00:00":
                    if str(current_position['date']) == "2022-01-21 21:30:00":
                        print("toto")


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
                print("LONG SL TP ", position["side"], " trade result: ", trade_result)
                return True
            else:
                return False
        elif position["side"] == "SHORT":
            trade_result = ((position['price'] - close_price) / position['price']) * leverage * 100
            if trade_result <= self.SL or trade_result >= self.TP:
                print("SHORT SL TP")
                return True
            else:
                return False
        return False

    def populate_sltp(self):
        df_tmp = self.df.copy()
        indexBuy = df_tmp[df_tmp['open_long_market'] == False].index
        df_tmp.drop(indexBuy, inplace=True)
        # df_tmp[df_tmp['open_long_market'] == True]

        # for tradeIndex in indexBuy:
        #     str_tradeIndex = 'open_long_market_' + str(tradeIndex)
        #     self.df[str_tradeIndex] = True

