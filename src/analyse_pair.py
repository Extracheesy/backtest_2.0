import pandas as pd
import numpy as np
import ta

class AnalysePair():
    def __init__(
        self,
        envelope_window,
        envelope_offset
    ):
        self.envelope_window = envelope_window
        self.envelope_offset = envelope_offset
        self.df_results = pd.DataFrame(columns=["symbol", "offset", "crossing_low", "crossing_high", "total_crossing"])

    def volatility_analyse_envelope_crossing(self, df, symbol):
        df_tmp = df.copy()
        df_tmp.drop(["volume"], axis=1, inplace=True)
        df_tmp["ma_base"] = ta.trend.SMAIndicator(close=df["close"], window=self.envelope_window).sma_indicator()
        df_tmp["envelope_high"] = df_tmp["ma_base"] + df_tmp["ma_base"] * self.envelope_offset / 100
        df_tmp["envelope_low"] = df_tmp["ma_base"] - df_tmp["ma_base"] * self.envelope_offset / 100
        df_tmp.dropna(inplace=True)

        df_tmp['volatility_high'] = np.where(df_tmp["high"] > df_tmp["envelope_high"], True, False)
        df_tmp['volatility_low'] = np.where(df_tmp["low"] < df_tmp["envelope_low"], True, False)
        df_tmp['volatility'] = df_tmp['volatility_high'] | df_tmp['volatility_low']
        vol_high = df_tmp['volatility_high'].sum()
        vol_low = df_tmp['volatility_low'].sum()
        vol = df_tmp['volatility'].sum()

        # print('high: ', vol_high, " low: ", vol_low, " all: ", vol)
        self.df_results.loc[len(self.df_results)] = [symbol.replace("/USDT", ""), self.envelope_offset, vol_low, vol_high, vol]

    def store_results(self):
        self.df_results.to_csv("envelope_" + str(self.envelope_offset) + "_analyse_volatility_results.csv")

