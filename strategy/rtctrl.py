import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime

# Class to real time control the strategy behaviours
class rtctrl():
    def __init__(self, current_datetime=None, params=None):
        self.lst_opening_type = []
        self.lst_closing_type = []

    def set_list_open_position_type(self, lst_opening_type):
        self.lst_opening_type = lst_opening_type

    def set_list_close_position_type(self, lst_closing_type):
        self.lst_closing_type = lst_closing_type

