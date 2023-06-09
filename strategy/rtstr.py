from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
import inspect
import importlib
import os
import ast
from os import path

class RealTimeStrategy(metaclass=ABCMeta):

    def __init__(self, params=None):
        self.lst_symbols = []
        self.SL = 0              # Stop Loss %
        self.TP = 0              # Take Profit %
        self.trigger_SL = False
        self.trigger_TP = False
        self.global_SL = 0       # Stop Loss % applicable to the overall portfolio
        self.global_TP = 0       # Take Profit % applicable to the overall portfolio
        self.trigger_global_TP = False
        self.trigger_global_SL = False
        self.safety_TP = 100
        self.safety_SL = -50
        self.liquidation_SL = 0
        self.trigger_liquidation_SL = False
        self.trailer_TP = 0
        self.trailer_delta = 0
        self.trigger_trailer = False
        self.trailer_global_TP = 0
        self.trailer_global_delta = 0
        self.trigger_global_trailer = False
        self.MAX_POSITION = 5    # Asset Overall Percent Size
        self.logger = None
        self.id = ""
        self.min_bol_spread = 0   # Bollinger Trend startegy
        self.trade_over_range_limits = False
        self.tradingview_condition = False
        self.short_and_long = False
        if params:
            self.MAX_POSITION = params.get("max_position", self.MAX_POSITION)
            if isinstance(self.MAX_POSITION, str):
                self.MAX_POSITION = int(self.MAX_POSITION)
            symbols = params.get("symbols", "")
            if symbols != "" and path.exists("./symbols/"+symbols):
                self.lst_symbols = pd.read_csv("./symbols/"+symbols)['symbol'].tolist()
            else:
                self.lst_symbols = symbols.split(",")

            self.SL = int(params.get("sl", self.SL))
            self.TP = int(params.get("tp", self.TP))
            self.global_SL = int(params.get("global_sl", self.global_SL))
            self.global_TP = int(params.get("global_tp", self.global_TP))
            self.logger = params.get("logger", self.logger)
            self.id = params.get("id", self.id)
            self.min_bol_spread = params.get("min_bol_spread", self.min_bol_spread)
            if isinstance(self.min_bol_spread, str):
                self.min_bol_spread = int(self.min_bol_spread)
            self.trade_over_range_limits = params.get("trade_over_range_limits", self.trade_over_range_limits)
            if isinstance(self.trade_over_range_limits, str):
                self.trade_over_range_limits = ast.literal_eval(self.trade_over_range_limits)
            self.tradingview_condition = params.get("tradingview_condition", self.tradingview_condition)
            if isinstance(self.tradingview_condition, str):
                self.tradingview_condition = ast.literal_eval(self.tradingview_condition)
            self.short_and_long = params.get("short_and_long", self.short_and_long)
            if isinstance(self.short_and_long, str):
                self.short_and_long = ast.literal_eval(self.short_and_long)
            self.trailer_TP = int(params.get("trailer_tp", self.trailer_TP))
            self.trailer_delta = int(params.get("trailer_delta", self.trailer_delta))
            self.trailer_global_TP = int(params.get("trailer_global_tp", self.trailer_global_TP))
            self.trailer_global_delta = int(params.get("trailer_global_delta", self.trailer_global_delta))
            self.liquidation_SL = int(params.get("liquidation_sl", self.liquidation_SL))

        self.trigger_trailer = self.trailer_TP > 0 and self.trailer_delta > 0
        self.trigger_global_trailer = self.trailer_global_TP > 0 and self.trailer_global_delta > 0
        self.trigger_SL = self.SL < 0
        self.trigger_TP = self.TP > 0
        self.trigger_global_SL = self.global_SL < 0
        self.trigger_global_TP = self.global_TP > 0
        self.trigger_liquidation_SL = self.liquidation_SL < 0

        self.rtctrl = None
        self.match_full_position = False # disabled

        self.str_short_long_position = StrOpenClosePosition()
        self.open_long = self.str_short_long_position.get_open_long()
        self.close_long = self.str_short_long_position.get_close_long()
        self.open_short = self.str_short_long_position.get_open_short()
        self.close_short = self.str_short_long_position.get_close_short()
        self.no_position = self.str_short_long_position.get_no_position()

        self.df_long_short_record = ShortLongPosition(self.lst_symbols, self.str_short_long_position)
        if self.trigger_trailer:
            self.df_trailer_TP = TrailerTP(self.lst_symbols, self.trailer_TP, self.trailer_delta)
        if self.trigger_global_trailer:
            self.df_trailer_global_TP = TrailerGlobalTP(self.trailer_global_TP, self.trailer_global_delta)

    def condition_for_opening_long_position(self, symbol):
        return False

    def condition_for_opening_short_position(self, symbol):
        return False

    def condition_for_closing_long_position(self, symbol):
        return False

    def condition_for_closing_short_position(self, symbol):
        return False

    def get_lst_opening_type(self):
        return [self.open_long, self.open_short]

    def get_lst_closing_type(self):
        return [self.close_long, self.close_short]

class StrOpenClosePosition():
    string = {
        "openlong" : 'OPEN_LONG',
        "openshort" : 'OPEN_SHORT',
        "closelong" : 'CLOSE_LONG',
        "closeshort" : 'CLOSE_SHORT',
        "noposition" : 'NO_POSITION'
    }

    def get_open_long(self):
        return self.string["openlong"]

    def get_open_short(self):
        return self.string["openshort"]

    def get_close_long(self):
        return self.string["closelong"]

    def get_close_short(self):
        return self.string["closeshort"]

    def get_no_position(self):
        return self.string["noposition"]
