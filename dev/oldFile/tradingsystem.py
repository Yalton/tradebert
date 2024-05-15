from alpaca_trade_api import REST
import threading
import requests
from datetime import datetime
from abc import ABC, abstractmethod
#import constants as const
import logging
import numpy as np
import time

# SEC_KEY = 'SEC_KEY'  # Enter Your Secret Key Here
# PUB_KEY = 'PUB_KEY'  # Enter Your Public Key Here
# # This is the base URL for paper trading
# BASE_URL = 'https://paper-api.alpaca.markets'

Log_Format = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename = "logs/logfile.log", filemode = "w", format = Log_Format,level = logging.DEBUG)
logger = logging.getLogger()

class AlpacaPaperSocket(REST):
    def __init__(self):
        super().__init__(
            key_id='YOUR_KEY_HERE',
            secret_key='YOUR_SECRET_KEY_HERE',
            base_url='https://paper-api.alpaca.markets'
        )

class TradingSystem(ABC):

    def __init__(self, api, symbol, time_frame, system_id, system_label):
        # Connect to api
        # Connect to BrokenPipeError
        # Save fields to class
        self.api = api
        self.symbol = symbol
        self.time_frame = time_frame
        self.system_id = system_id
        self.system_label = system_label
        thread = threading.Thread(target=self.system_loop)
        thread.start()

    @abstractmethod
    def analyze_asset(self):
        pass

    @abstractmethod
    def market_buy(self):
        pass

    @abstractmethod
    def market_sell(self):
        pass
    
    @abstractmethod
    def limit_buy(self):
        pass
    
    @abstractmethod
    def limit_sell(self): 
        pass

    @abstractmethod
    def system_loop(self):
        pass