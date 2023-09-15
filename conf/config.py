from utilities.utils import add_n_columns_to_list

GET_DATA = True
RUN_BACKTEST = True
RUN_BENCHMARK = True
RUN_FILTER = True
GLOBAL_FILTER = True

lst_symbol_ALL = [
    'BTC', 'ETH', 'XRP', 'EOS', 'BCH', 'LTC', 'ADA', 'ETC', 'LINK', 'TRX', 'DOT', 'DOGE', 'SOL', 'MATIC', 'BNB', 'UNI',
    'ICP', 'AAVE', 'FIL', 'XLM', 'ATOM',
    'XTZ', 'SUSHI', 'AXS', 'THETA', 'AVAX', 'DASH', 'SHIB', 'MANA', 'GALA', 'SAND', 'DYDX', 'CRV', 'NEAR', 'EGLD', 'KSM',
    'AR', 'REN', 'FTM', 'PEOPLE', 'LRC', 'NEO', 'ALICE', 'WAVES', 'ALGO',
    'IOTA', 'YFI', 'ENJ', 'GMT', 'ZIL', 'IOST', 'APE', 'RUNE', 'KNC', 'APT', 'CHZ', 'XMR', 'ROSE', 'ZRX', 'KAVA',
    'ENS', 'GAL', 'MTL', 'AUDIO', 'SXP', 'C98', 'OP', 'RSR', 'SNX', 'STORJ', '1INCH', 'COMP', 'IMX', 'LUNA2', 'FLOW',
    'REEF', 'TRB', 'QTUM', 'API3', 'MASK', 'WOO', 'GRT', 'BAND', 'STG', 'LUNC', 'ONE', 'JASMY', 'FOOTBALL', 'MKR',
    'BAT', 'MAGIC', 'ALPHA', 'LDO', 'OCEAN', 'CELO', 'BLUR', 'MINA',
    # 'CORE',
    'CFX', 'HIGH', 'ASTR', 'AGIX', 'GMX',
    'LINA', 'ANKR',
    # 'GFT',
    'ACH', 'FET', 'FXS', 'RNDR', 'HOOK', 'BNX', 'SSV',
    # 'BGHOT10',
    'LQTY', 'STX',
    # 'TRU',
    'DUSK', 'HBAR', 'INJ', 'BEL', 'COTI', 'VET', 'ARB', 'TOMO',
    # 'LOOKS',
    'KLAY', 'FLM', 'OMG', 'RLC', 'CKB', 'ID',
    'LIT', 'JOE', 'TLM', 'HOT', 'BLZ', 'CHR', 'RDNT', 'ICX', 'HFT', 'ONT', 'ZEC', 'UNFI', 'NKN', 'ARPA', 'DAR', 'SFP',
    'CTSI', 'SKL', 'RVN', 'CELR', 'FLOKI', 'SPELL', 'SUI', 'EDU', 'PEPE',
    # 'METAHOT',
    'IOTX', 'CTK', 'STMX', 'UMA',
    # 'BSV',
    # '10000AIDOGE',
    # '10000LADYS',
    # 'TON',
    'GTC', 'DENT', 'ZEN', 'PHB',
    # 'ORDI',
    'KEY', 'IDEX', 'SLP']

lst_symbol_BITGET = ['BTC', 'ETH', 'XRP', 'EOS', 'BCH', 'LTC', 'ADA', 'ETC', 'LINK', 'TRX', 'DOT', 'DOGE', 'SOL',
                     'MATIC', 'BNB', 'UNI', 'ICP', 'AAVE', 'FIL', 'XLM', 'ATOM', 'XTZ', 'SUSHI', 'AXS', 'THETA',
                     'AVAX', 'DASH', 'SHIB', 'MANA', 'GALA', 'SAND', 'DYDX', 'CRV', 'NEAR', 'EGLD', 'KSM', 'AR',
                     'REN', 'FTM', 'PEOPLE', 'LRC', 'NEO', 'ALICE', 'WAVES', 'ALGO', 'IOTA', 'YFI', 'ENJ', 'GMT',
                     'ZIL', 'IOST', 'APE', 'RUNE', 'KNC', 'APT', 'CHZ', 'XMR', 'ROSE', 'ZRX', 'KAVA', 'ENS', 'GAL',
                     'MTL', 'AUDIO', 'SXP', 'C98', 'OP', 'RSR', 'SNX', 'STORJ', '1INCH', 'COMP', 'IMX', 'LUNA2',
                     'FLOW', 'REEF', 'TRB', 'QTUM', 'API3', 'MASK', 'WOO', 'GRT', 'BAND', 'STG', 'LUNC', 'ONE',
                     'JASMY', 'FOOTBALL', 'MKR', 'BAT', 'MAGIC', 'ALPHA', 'LDO', 'OCEAN', 'CELO', 'BLUR', 'MINA',
                     'CFX', 'HIGH', 'ASTR', 'AGIX', 'GMX', 'LINA', 'ANKR', 'ACH', 'FET', 'FXS',
                     'RNDR', 'HOOK', 'BNX', 'SSV', 'LQTY', 'STX', 'TRU', 'DUSK', 'HBAR', 'INJ',
                     'BEL', 'COTI', 'VET', 'ARB', 'TOMO', 'KLAY', 'FLM', 'OMG', 'RLC', 'CKB', 'ID', 'LIT',
                     'JOE', 'TLM', 'HOT', 'BLZ', 'CHR', 'RDNT', 'ICX', 'HFT', 'ONT', 'ZEC', 'UNFI', 'NKN', 'ARPA',
                     'DAR', 'SFP', 'CTSI', 'SKL', 'RVN', 'CELR', 'FLOKI', 'SPELL', 'SUI', 'EDU',
                     'IOTX', 'CTK', 'STMX', 'UMA', 'GTC', 'DENT', 'ZEN',
                     'PHB', 'KEY', 'IDEX', 'SLP', 'COMBO', 'AMB', 'LEVER', 'RAD', 'ANT', 'QNT', 'MAV',
                     'MDT', 'XVG']

lst_symbol_BTC_ETH = ['BTC', 'ETH']
lst_symbol_BTC = ['BTC']
lst_symbol_ETH = ['ETH']

VOLATILITY_ANALYSE = False
ENGAGED_OVERLAP = False
PRINT_OUT = False
NO_WARNINGS = True

NB_TOP_PERFORMER = 5
lst_performer = ["final_wallet", "sharpe_ratio",
                 # "max_trades_drawdown",
                 "vs_hold_pct", "global_win_rate"]

# lst_stop_loss = [0, -2, -5, -7, -10]
lst_stop_loss = [0]
lst_offset = [2, 3, 4, 5, 6]

"""
lst_bol_window = [20, 50, 100]
lst_bol_std = [2.0, 2.25, 2.5]
lst_min_bol_spread = [0]
lst_long_ma_window = [20, 50, 100, 200, 500]
"""

lst_bol_window = [10, 20, 30]
lst_bol_std = [2.0, 2.25, 2.5]
lst_min_bol_spread = [0]
lst_long_ma_window = [10, 20, 50, 100]

lst_rsi_high = [0, 70, 50]
lst_rsi_low = [30, 50, 100]

lst_stochOverBought = [0.6, 0.7, 0.8, 0.9]
lst_stochOverSold = [0.1, 0.2, 0.3, 0.4]
lst_willOverSold = [-60, -70, -80, -90]
lst_willOverBought = [-40, -30, -20, -10]

dct_lst_param = {
    "lst_stop_loss": lst_stop_loss,
    "lst_offset": lst_offset,
    "lst_bol_window": lst_bol_window,
    "lst_bol_std": lst_bol_std,
    "lst_min_bol_spread": lst_min_bol_spread,
    "lst_long_ma_window": lst_long_ma_window,
    "lst_rsi_high": lst_rsi_high,
    "lst_rsi_low": lst_rsi_low,
    "lst_stochOverBought": lst_stochOverBought,
    "lst_stochOverSold": lst_stochOverSold,
    "lst_willOverSold": lst_willOverSold,
    "lst_willOverBought": lst_willOverBought,
}

tf = "1h"
start = "2023-01-01 00:00:00"

# symbol = "ALL"
symbol = "BITGET"
# symbol = "BTC_ETH"
# symbol = "ETH"
# symbol = "BTC"

# lst_strategy = ["bollinger_reversion"]
# lst_strategy = ["bol_trend", "bollinger_reversion"]
# lst_strategy = ["bol_trend", "big_will", "bollinger_reversion"]
# lst_strategy = ["bol_trend", "big_will"]
lst_strategy = ["big_will"]
# lst_strategy = ["bol_trend"]
# lst_strategy = ["bol_trend_no_ma"]
# lst_strategy = ["bol_trend", "bol_trend_no_ma", "big_will"]

# lst_type=["short"]
# lst_type=["long"]
lst_type = ["long", "short"]

# lst_filter_start = ["2021"]
# lst_filter_start = ["2022"]
# lst_filter_start = ["2023", "1M"]
# lst_filter_start = ["2023", "1M", "2W"]
lst_filter_start = ["2W", "1M"]

RUN_ON_INTERVALS = True
INTERVALS = 10 + 1
dct_inetrvals = {}

MULTI_PROCESS = True
MULTI_THREAD = False

COLAB = False
COLAB_DIR_ROOT = "../drive/MyDrive/Colab Notebooks/Backtest/"

QUANTILE = 0.9 # 10%
# QUANTILE = 0.8 # 20%

from src.activity_tracker import ActivityTracker
TRACKER = ActivityTracker()

LST_COLUMN_STRATEGY_BENCHMARK= [
    "start_date",
    "startegy",
    "pair",
    "max_final_wallet",
    "min_final_wallet",
    "mean_final_wallet",
    "%_>_1000_final_wallet",
    "%_>_1500_final_wallet",
    "%_>_2000_final_wallet",
    "%_>_2500_final_wallet",
    "%_>_3500_final_wallet",
    "max_sharpe_ratio",
    "min_sharpe_ratio",
    "mean_sharpe_ratio",
    "%_>_1_sharpe_ratio",
    "%_>_2_sharpe_ratio",
    "%_>_3_sharpe_ratio",
    "max_vs_hold_pct",
    "min_vs_hold_pct",
    "mean_vs_hold_pct",
    "max_global_win_rate",
    "min_global_win_rate",
    "mean_global_win_rate",
    "quantile_10",
    "quantile_20",
    "quantile_30",
    "quantile_40",
    "quantile_50",
    "quantile_60",
    "quantile_70",
    "quantile_80",
    "quantile_90"
]

lst_header_parameters = [
    "start_date",
    "strategy",
    "score"
]

LST_HEADERS_PAIRS_BENCMARK = [
    "start_date",
    "strategy",
    "pair",
    "score"
]

LST_HEADERS_LST_PAIRS_COMPARE_BENCHMARK = [
    "pair"
]

lst_paramters = [
    "stop_loss",
    "offset",
    "bol_window",
    "bol_std",
    "min_bol_spread",
    "long_ma_window",
    "stochOverBought",
    "stochOverSold",
    "willOverSold",
    "willOverBought",
    "rsi_high",
    "rsi_low"
]

LST_COLUMN_PARAMETER_BENCHMARK = lst_header_parameters + lst_paramters
NB_PAIRS_SELECTED = 35
LST_COLUMN_PARAMETER_BENCHMARK = add_n_columns_to_list(LST_COLUMN_PARAMETER_BENCHMARK, "pair_", NB_PAIRS_SELECTED)

final_target_results = "merged_results"


