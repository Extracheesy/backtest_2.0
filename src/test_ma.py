import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import mplfinance as mpf

df = pd.read_csv("test_ma.csv")

df["date"] = pd.to_datetime(df["timestamp"])

df.reset_index(inplace=True, drop=True)

df.set_index('date', inplace=True)


# add multiple additional data sets
apdict = mpf.make_addplot(df[['ma_base']])

mpf.plot(df,type='candle',mav=(5, 3),volume=False, title='BTC', addplot=apdict)
