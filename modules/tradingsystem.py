import threading
import configparser
from abc import ABC, abstractmethod
import alpaca_trade_api
import ccxt

import math
import numpy as np
import configparser
import pandas as pd

from alpaca_trade_api.rest import TimeFrame
import datetime
from time import sleep
import logging

from modules import tech_indicators as TechInd
from .fetch_data import DataAggregator
from modules  import TradingSrategies as TradeStrats
from modules import databaseInterface as DBMGR
from modules import constants
#import asset_analysis as assetAnalysis

# from kucoin.client import Market as KucoinMarket
# from kucoin.client import Trade as KucoinTrade
# #from kucoin.client import KucoinUser # Problem 

# from kucoin_futures.client import Market as KucoinFuturesMarket
# from kucoin_futures.client import Trade as KucoinFuturesTrade



def initializeLogger(logName): 
    Log_Format = "%(levelname)s %(asctime)s - %(message)s"
    logging.basicConfig(filename = f"logs/{logName}logfile.log", filemode = "a", format = Log_Format,level = logging.DEBUG)
    return logging.getLogger()




# Abstract Parent class
class TradingSystem(ABC):

    # Dictionary to Map strings to Trading Strategy functions from the TradeStrats file
    strategy_dict = {
        'TrendFollowing': TradeStrats.TrendFollowingStrategy,
        'MomentumTrading': TradeStrats.MomentumTradingStrategy,
        'ReversalTrading': TradeStrats.ReversalTradingStrategy,
        'VolumeAnalysis': TradeStrats.VolumeAnalysisStrategy,
        'BreakoutTrading': TradeStrats.BreakoutTradingStrategy,
        'TrendStrengthVolatility': TradeStrats.TrendStrengthVolatilityStrategy,
        'VolumeFlow': TradeStrats.VolumeFlowStrategy,
        'SupportResistance': TradeStrats.SupportResistanceStrategy,
        'TrendContinuationReversal': TradeStrats.TrendContinuationReversalStrategy,
        'MeanReversion': TradeStrats.MeanReversionStrategy,
        'BollingerBands': TradeStrats.BollingerBandsStrategy,
        'MACD': TradeStrats.MACDStrategy,
        'SqueezeMomentum': TradeStrats.SqueezeMomentumStrategy,
        'CryptoLadder': TradeStrats.CryptoLadderStrategy
    }

    """
        Base Trading system class
        =========================
        All other classes inherit from this Abstract Parent class
        Cannot place any orders on it's own, but it provides member vars and common functions for other classes 
        Member Vars: 
        =========================
        system_id: Numerical ID of trading system
        system_label: String Label for trading system 
        symbols: Symbols of assets user desires the system to trade 
        strategy_weights: Dictionary of Trading strategies and the Weights that should be applied to said strategies 
        congruence_level: Congruence Level of trading strategies i.e How much should the individual strategies agree on a signal before it is made 
            - Accepted Values are low, medium, and high 
        risk_reward_ratio: What should the risk to reward ratio be for each trade (Primarily determines how stops and targets are calculated)
        timeframe: What timeframe of data should be looked at; defaults to a 15 Minute chart
    """
    def __init__(self, system_id, system_label, brokerage_id, symbols, strategy_weights, congruence_level="medium", risk_reward_ratio=1, timeframe='15m'):

        self.system_id = system_id
        self.system_label = system_label

        self.brokerage_id = brokerage_id

        self.symbols = symbols
        self.timeframe = timeframe
        self.strategy_weights = strategy_weights

        self.signals_dict = {}
        self.strategies = []

        self.congruence_level = congruence_level
        self.risk_reward_ratio = risk_reward_ratio

        self.config = self.read_config()

        self.DB = DBMGR.DatabaseManager(host=self.config.get("sql", "host"), database=self.config.get("sql", "database"), user=self.config.get("sql", "user"), password=self.config.get("sql", "password"))

        self.logger = initializeLogger(system_label)

        self.thread = threading.Thread(target=self.system_loop)
        
    # Read from Config file and import api keys
    def read_config(self):
        config = configparser.ConfigParser()
        config.read("config.ini")
        return config

    # Function to handle acquiring historical data
    def acquire_historical_data(self, symbol): 
        aggregator = DataAggregator(self.config.get("alphavantage", "api_key"))
        start_date = datetime.datetime.now() - datetime.timedelta(days=50)
        end_date = datetime.datetime.now()
        data = aggregator.fetch_stock_data(symbol, start_date, end_date, interval=self.timeframe)
        data.columns = data.columns.str.strip()
        return data

    # Function to convert given timeframe to seconds for sleeping in trading loop
    def timeframe_to_seconds(self, timeframe):
        timeframe_value = int(timeframe[:-1])  # get the numerical part
        timeframe_unit = timeframe[-1]  # get the last character which is the unit

        if timeframe_unit == 's':
            return timeframe_value # For debugging
        elif timeframe_unit == 'm':
            return timeframe_value * 60  # convert minutes to seconds
        elif timeframe_unit == 'h':
            return timeframe_value * 60 * 60  # convert hours to seconds
        elif timeframe_unit == 'd':
            return timeframe_value * 60 * 60 * 24  # convert days to seconds
        elif timeframe_unit == 'w':
            return timeframe_value * 60 * 60 * 24 * 7  # convert weeks to seconds
        elif timeframe_unit == 'M':
            return timeframe_value * 60 * 60 * 24 * 30  # convert months to seconds (approx)
        else:
            raise ValueError(f'Invalid timeframe unit: {timeframe_unit}')

    #####################
    # DATABASE MANAGEMENT
    #####################

    def initializeDB(self):
        # Create a database connection
        self.DB.create_connection()

        # Load external SQL script
        script_path = 'databaseInit.sql'
        with open(script_path, 'r') as file:
            script_content = file.read()

        # Split the script content into individual queries
        queries = script_content.split(';')

        # Execute queries
        for query in queries:
            self.DB.execute_query(query)

        # print(f"Executing {constants.BROKERAGE_TABLE_INIT_QUERY}")

        self.DB.execute_query(constants.BROKERAGE_TABLE_INIT_QUERY)

        insert_table_sql = """
            INSERT INTO SystemTable (System_ID, System_Name, Brokerage_ID)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE System_Name = VALUES(System_Name);
            """
        data_tuple = (self.system_id, self.system_label, self.brokerage_id)

        self.DB.insert_into_table(insert_table_sql, data_tuple)

        for symbol in self.symbols: 
            insert_table_sql = """
                INSERT IGNORE INTO Asset (System_ID, Symbol, Tradable, Asset_Class)
                VALUES (%s, %s, %s, %s);
                """
            data_tuple = (self.system_id, symbol, self.get_tradeable(symbol), self.get_asset_class(symbol))

            self.DB.insert_into_table(insert_table_sql, data_tuple)

        for symbol in self.symbols: 
            # Get the Asset_ID for the symbol and system
            asset_id = self.get_asset_id(symbol)

            # If Asset_ID is found, insert the strategy weights
            if asset_id:
                # print(self.strategy_weights)
                weights = self.strategy_weights[symbol]
                # Prepare SQL query
                insert_table_sql = """
                    INSERT INTO StrategyWeights (
                        Asset_ID, TrendFollowing, MomentumTrading, ReversalTrading, VolumeAnalysis, 
                        BreakoutTrading, TrendStrengthVolatility, VolumeFlow, SupportResistance, 
                        TrendContinuationReversal, MeanReversion, BollingerBands, MACD, 
                        SqueezeMomentum, CryptoLadder
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        TrendFollowing = VALUES(TrendFollowing), 
                        MomentumTrading = VALUES(MomentumTrading), 
                        ReversalTrading = VALUES(ReversalTrading),
                        VolumeAnalysis = VALUES(VolumeAnalysis),
                        BreakoutTrading = VALUES(BreakoutTrading),
                        TrendStrengthVolatility = VALUES(TrendStrengthVolatility),
                        VolumeFlow = VALUES(VolumeFlow),
                        SupportResistance = VALUES(SupportResistance),
                        TrendContinuationReversal = VALUES(TrendContinuationReversal),
                        MeanReversion = VALUES(MeanReversion),
                        BollingerBands = VALUES(BollingerBands),
                        MACD = VALUES(MACD),
                        SqueezeMomentum = VALUES(SqueezeMomentum),
                        CryptoLadder = VALUES(CryptoLadder);
                """
                # Prepare data
                data_tuple = (asset_id, weights['TrendFollowing'], weights['MomentumTrading'], weights['ReversalTrading'], 
                            weights['VolumeAnalysis'], weights['BreakoutTrading'], weights['TrendStrengthVolatility'], 
                            weights['VolumeFlow'], weights['SupportResistance'], weights['TrendContinuationReversal'], 
                            weights['MeanReversion'], weights['BollingerBands'], weights['MACD'], 
                            weights['SqueezeMomentum'], weights['CryptoLadder'])

                # Execute the query
                self.DB.insert_into_table(insert_table_sql, data_tuple)
            else:
                print(f"No Asset_ID found for System_ID {self.system_id} and Symbol {symbol}")
        self.DB.close_connection()

    def get_asset_id(self, symbol):
        # Prepare SQL query
        select_query = """
            SELECT Asset_ID FROM Asset WHERE System_ID = %s AND Symbol = %s;
        """
        # Prepare data
        data_tuple = (self.system_id, symbol)
        # Execute the query
        rows = self.DB.select_from_table(select_query, data_tuple)
        # If the result is not empty, return the Asset_ID
        if rows:
            return rows[0][0]  # assuming the first column is Asset_ID
        else:
            return None

    #def insert_strategy_weights_for_symbol(self, system_id, symbol, weights):


    #####################
    # STRATEGY MANAGEMENT
    #####################
    def add_strategy(self, strategy_name, data, weight):
        StrategyClass = self.strategy_dict[strategy_name]
        self.strategies.append(StrategyClass(data=data, weight=weight))

    def remove_strategy(self, strategy_name):
        self.strategies = [s for s in self.strategies if s.__class__.__name__ != strategy_name]

    def set_congruence_level(self, level):
        self.congruence_level = level

    ##########################
    # ASSET ANALYSIS & SIGGEN
    ##########################
    def generate_ensemble_signals(self):
        # Define threshold levels
        threshold_levels = {
            'low': {'buy': 0.3, 'strong_buy': 0.6, 'sell': -0.3, 'strong_sell': -0.6},
            'medium': {'buy': 0.5, 'strong_buy': 0.75, 'sell': -0.5, 'strong_sell': -0.75},
            'high': {'buy': 0.7, 'strong_buy': 0.9, 'sell': -0.7, 'strong_sell': -0.9}
        }
        
        thresholds = threshold_levels[self.congruence_level]

        # Create a 2D numpy array with signals from each strategy
        # for strategy in self.strategies:
        #     print(strategy.generate_signal())

        signals_array = np.stack([strategy.generate_signal() for strategy in self.strategies])

        # Sum the signals for each period (i.e., along the strategy axis)
        total_signals = signals_array.sum(axis=0)

        weight = sum(strategy.weight for strategy in self.strategies)

        #print(weight)

        # Normalize total_signals by sum of weights
        total_signals = total_signals / sum(strategy.weight for strategy in self.strategies)

        # Apply thresholds to total_signals
        total_signals = np.where(total_signals > thresholds['strong_buy'], "Strong Buy", 
                                np.where(total_signals > thresholds['buy'], "Buy",
                                        np.where(total_signals < thresholds['sell'], "Sell",
                                                np.where(total_signals < thresholds['strong_sell'], "Strong Sell", "Hold"))))
        # Convert the signal array into a pandas Series with the same index as df
        df = self.strategies[-1].get_dataframe()
        signal_series = pd.Series(total_signals, index=df.index)

        # Now create a new DataFrame by merging the original df and the signal series
        df_signal = pd.DataFrame({'Date': df.index, 'Signal': signal_series}).set_index('Date')

        return df_signal

    def analyze_assets(self):
        for symbol in self.symbols:
            # Reset strategies at the start of each iteration
            self.strategies = []

            data = self.acquire_historical_data(symbol)

            # if(data.is_empty()): 
            #     print("No stock Data")
            #     break 

            # Loop over strategies and their weights
            for strategy_name, weight in self.strategy_weights[symbol].items():
                print("Adding ", strategy_name, " with weight ", weight)
                self.add_strategy(strategy_name, data.copy(), weight)

            signals = self.generate_ensemble_signals()

            # Store the signals in the dictionary
            self.signals_dict[symbol] = signals
            self.signals_dict[symbol].to_csv(f"csvs/screened/{symbol}_screened.csv")
            self.logger.debug(f"Performed Analysis for: {symbol}")

        return self.signals_dict

    @abstractmethod
    def get_balance(self): 
        pass 

    def get_current_price(self, symbol):
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

    ###################
    # TRADE CALCULATION
    ###################
    def calculate_trade_amount(self, signal):
        portfolio_balance = float(self.get_balance())  # assuming this function exists in your system
        if signal == "Buy":
            return math.floor(portfolio_balance * 0.2)
        elif signal == "Strong Buy":
            return math.floor(portfolio_balance * 0.3)

    def calculate_stop_price(self, symbol):
        #atr = self.get_atr(symbol)  # assuming this function exists in your system
        current_price = self.get_current_price(symbol)  # assuming this function exists in your system
        
        return math.floor(current_price * 0.95)

    def calculate_take_profit_price(self, symbol):
        current_price = self.get_current_price(symbol)  # assuming this function exists in your system
        price_change = 1 + (0.05 * self.risk_reward_ratio)
        return math.floor(current_price * price_change)

        #portfolio_balance = float(portfolio_balance)
        #print(f"In Calc_trade_ammount {portfolio_balance}")
        #return current_price - atr

class AlpacaSystem(TradingSystem):
    
    """
        Alpaca Trading System class
        ===========================
        This class is a specific implementation of the TradingSystem abstract base class, 
        built to work with the Alpaca trading API. The AlpacaSystem uses the configuration,
        member variables, and common functions defined in the TradingSystem class to manage 
        trading operations, while adding its specific logic for interfacing with Alpaca's API.

        Member Variables:
        ===========================
        system_id: Numerical ID of the trading system
        system_label: String Label for the trading system 
        symbols: Symbols of assets the user desires the system to trade 
        strategy_weights: Dictionary of trading strategies and the weights applied to each strategy
        congruence_level: Congruence level of trading strategies - how much should the individual strategies agree on a signal before it is made. Accepted values are low, medium, and high
        risk_reward_ratio: Desired risk-to-reward ratio for each trade (mainly used to determine stop loss and target levels)
        timeframe: Timeframe of data to analyze; defaults to a 15-minute chart
        config: Configuration object containing Alpaca API credentials
        api: Alpaca REST API client object
    """
    def __init__(self, system_id, system_label, symbols, strategy_weights, brokerage_id=1, congruence_level="medium", risk_reward_ratio=1, timeframe='15m'):
        super().__init__(system_id=system_id, system_label=system_label, brokerage_id=brokerage_id, symbols=symbols, strategy_weights=strategy_weights, congruence_level=congruence_level, risk_reward_ratio=risk_reward_ratio, timeframe=timeframe)

        # For real trading, don't enter a base_url
        BASE_URL = 'https://paper-api.alpaca.markets'
        self.api = alpaca_trade_api.REST(key_id=self.config.get("Alpaca", "api_key"), secret_key=self.config.get("Alpaca", "api_secret"), base_url=BASE_URL, api_version='v2')
        self.logger.debug(f"AlpacaSystem: initialized with ID: {self.system_id}, Label:{self.system_label}")
        self.initializeDB()


    def get_balance(self):
        self.account = self.api.get_account()
        # Check if account is active
        if self.account.status == 'ACTIVE':
            # Check account balance
            self.cash = self.account.cash
            self.portfolio_value = self.account.portfolio_value
            self.buying_power = self.account.buying_power
            
            print(f'Cash: {self.cash}')
            print(f'Portfolio Value: {self.portfolio_value}')
            print(f'Buying Power: {self.buying_power}')
            return self.buying_power
        else:
            print("Account is not active.")

    def get_current_price(self, symbol):
        timeframe = '1Min' 
        bars = self.api.get_bars(symbol, timeframe, limit=1)
        last_bar = bars[-1]
        return last_bar.c

    def get_asset_class(self, symbol):
        # Retrieve the asset information
        asset_info = self.api.get_asset(symbol)
        
        # Get the class of the asset using getattr
        asset_class = getattr(asset_info, 'class')
        
        # Check the class of the asset
        if asset_class == 'us_equity':
            return 'stock'
        elif asset_class == 'crypto':
            return 'cryptocurrency'
        else:
            return 'unknown'

    def get_tradeable(self, symbol): 
        return self.api.get_asset(symbol).tradable

    def get_share_count(self, symbol):
        try:
            position = self.api.get_position(symbol)
            return position.qty
        except Exception as e:
            print(f"Could not get position for symbol {symbol}. Error: {e}")
            self.logger.debug(f"AlpacaSystem:: Could not get position for symbol {symbol}. Error: {e}")
            return 0

    def cancel_all_orders_for_asset(self, symbol):
        # Get a list of all open orders
        orders = self.api.list_orders(status='open')

        # Iterate over the orders
        for order in orders:
            # If the order's symbol matches the one we're interested in
            if order.symbol == symbol:
                # Cancel the order
                self.api.cancel_order(order.id)

    def market_buy(self, symbol, qty):
        try: 
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            self.logger.debug(f"Market Buy order sent for {qty} of symbol: {symbol}")
            return order
        except Exception as e:  
            self.logger.debug(f"EXCEPTION Market Buy order failed for {qty} of symbol: {symbol}. Exception: {str(e)}")

    def market_sell(self, symbol, qty):
        try: 
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            self.logger.debug(f"Market Sell order sent for {qty} of symbol: {symbol}")
            return order
        except Exception as e:  
            self.logger.debug(f"EXCEPTION Market Sell order failed for {qty} of symbol: {symbol}. Exception: {str(e)}")

    def limit_buy(self, symbol, qty, price):
        order = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side='buy',
            type='limit',
            time_in_force='gtc',
            limit_price=price
        )
        self.logger.debug(f"Limit Buy order sent for {qty} of symbol: {symbol} at a price of {price}")
        return order

    def limit_sell(self, symbol, qty, price):
        order = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type='limit',
            time_in_force='gtc',
            limit_price=price
        )
        self.logger.debug(f"Limit Sell order sent for {qty} of symbol: {symbol} at a price of {price}")
        return order
    
    def stop_order(self, symbol, qty, stop_price): 
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                type='stop',
                time_in_force='gtc',
                stop_price=stop_price,
            )
            return order
        except Exception as e: 
            self.logger.debug(f"EXCEPTION Stop order failed for {qty} of symbol: {symbol} at a stop price of {stop_price}. Exception: {str(e)}")

    def stop_limit_order(self, symbol, qty, stop_price, limit_price, side): 
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='stop_limit',
                time_in_force='gtc',
                stop_price=stop_price,
                limit_price=limit_price,
            )
            self.logger.debug(f"Stop Limit {side} order sent for {qty} of symbol: {symbol}. Stop price: {stop_price}, Limit Price {limit_price}")
            return order
        except Exception as e: 
            self.logger.debug(f"EXCEPTION Stop limit order failed for {qty} of symbol: {symbol} with stop price {stop_price} and limit price {limit_price}. Exception: {str(e)}")

    def trailing_stop_order(self, symbol, qty, trail_percent): 
        try: 
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                type='trailing_stop',
                time_in_force='gtc',
                trail_percent=trail_percent,
            )
            self.logger.debug(f"Trailing stop order sent for {qty} of symbol: {symbol} with trail percent {trail_percent}")
            return order
        except Exception as e: 
            self.logger.debug(f"EXCEPTION Trailing stop order failed for {qty} of symbol: {symbol} with trail percent {trail_percent}. Exception: {str(e)}")
    
    def bracket_order(self, symbol, qty, symbol_price): 
        try: 
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='gtc',
                order_class='bracket',
                stop_loss={'stop_price': math.floor(symbol_price * 0.95),
                        'limit_price':  math.floor(symbol_price * 0.94)},
                take_profit={'limit_price': math.floor(symbol_price * (1 + (0.05 * self.risk_reward_ratio)))}
            )

            self.logger.debug(f"Bracket order sent for {qty} of symbol: {symbol}.")
        except Exception as e: 
            self.logger.debug(f"EXCEPTION Bracket order failed for {qty} of symbol: {symbol}. Exception: {str(e)}")
    
    def system_loop(self):
        while True:
            bought_assets = set()  # Stores the assets that have been bought and not sold yet
            portfolio = self.api.list_positions()
            for position in portfolio:
                print("{} shares of {}".format(position.qty, position.symbol))
                self.logger.debug(f"Portfolio is holding {position.qty} of symbol: {position.symbol}")
                bought_assets.add(position.symbol)

            self.analyze_assets() # Perform analysis on all assets system is designated to trade
            clock = self.api.get_clock()
            now = datetime.datetime.now()
            self.logger.debug(f"AlpacaSystem: Trading started at {now} on chart interval {self.timeframe}")
            for symbol, signal_df in self.signals_dict.items():
                # asset = self.api.get_asset(symbol)
                if self.get_tradeable(symbol):
                    # Only Perform trade if symbol is a stock and the market is open, or if symbol is a cryptocurrency 
                    if((self.get_asset_class(symbol) == "stock" and clock.is_open) or self.get_asset_class(symbol) == "cryptocurrency"):
                        signal = signal_df.iloc[-1]['Signal']  # get the latest signal
                        if signal in ["Buy", "Strong Buy"] and symbol not in bought_assets:
                            
                            # Calculate amount of asset to purchase
                            trade_capital = self.calculate_trade_amount(signal)
                            qty = math.floor(trade_capital / self.get_current_price(symbol))

                            self.bracket_order(symbol, qty, self.get_current_price(symbol))

                            bought_assets.add(symbol) # Check if stop limit order goes through; remove asset from list if so 
                        
                        elif signal in ["Sell", "Strong Sell"]:
                            qty = self.get_share_count(symbol)
                            if(qty): 
                                trade_capital = self.calculate_trade_amount(signal)
                                self.cancel_all_orders_for_asset(symbol) # Cancel all orders for assets
                                bought_assets.remove(symbol)  # remove symbol from bought assets set

                                # If order is Sell, sell of half of assets
                                if signal in ["Sell"]:
                                    print(f"self.market_sell({symbol}, {(qty * 0.5)})")
                                    self.market_sell(symbol, (qty * 0.5))

                                # If order is Strong Sell, dump all of current asset
                                elif signal in ["Strong Sell"]: 
                                    print(f"self.market_sell({symbol}, {qty})")
                                    self.market_sell(symbol, qty)
                            else: 
                                self.logger.debug(f"AlpacaSystem: Sell Order received for {signal}, but portfolio is not holding any")
                else: 
                    self.symbols.remove(symbol)
                    self.logger.debug(f"AlpacaSystem: Unable to trade {symbol}, removing from symbols list")

            print(f"Sleeping for {self.timeframe_to_seconds(self.timeframe)} S")
            self.logger.debug(f"AlpacaSystem: Completed trading for timeframe, sleeping for {self.timeframe}")
            sleep(self.timeframe_to_seconds(self.timeframe))# wait some time before trading this symbol again
                            
# class BinanceSystem(TradingSystem):
    
#     """
#         Binance Trading System class
#         ===========================
#         This class is a specific implementation of the TradingSystem abstract base class, 
#         built to work with the Binance trading API. The BinanceSystem uses the configuration,
#         member variables, and common functions defined in the TradingSystem class to manage 
#         trading operations, while adding its specific logic for interfacing with Binance's API.

#         Member Variables:
#         ===========================
#         system_id: Numerical ID of the trading system
#         system_label: String Label for the trading system 
#         symbols: Symbols of assets the user desires the system to trade 
#         strategy_weights: Dictionary of trading strategies and the weights applied to each strategy
#         congruence_level: Congruence level of trading strategies - how much should the individual strategies agree on a signal before it is made. Accepted values are low, medium, and high
#         risk_reward_ratio: Desired risk-to-reward ratio for each trade (mainly used to determine stop loss and target levels)
#         timeframe: Timeframe of data to analyze; defaults to a 15-minute chart
#         config: Configuration object containing Binance API credentials
#         binance: Binance API client object
#     """

#     def __init__(self, system_id, system_label, symbols, strategy_weights, brokerage_id=2, congruence_level="medium", risk_reward_ratio=1, timeframe='15m'):
#         super().__init__(system_id=system_id, system_label=system_label, brokerage_id=brokerage_id, symbols=symbols, strategy_weights=strategy_weights, congruence_level=congruence_level, risk_reward_ratio=risk_reward_ratio, timeframe=timeframe)
#         # Initialize Binance client
#         self.brokerage_id = 2
#         self.binance = ccxt.binance({
#             'apiKey': self.config.get("Binance", "api_key"),
#             'secret': self.config.get("Binance", "api_secret"),
#         })
#         self.logger.debug(f"BinanceSystem initialized with ID: {self.system_id}, Label:{self.system_label}")


#     # Function to perform market buy
#     def market_buy(symbol, amount):
#         order = self.binance.create_market_buy_order(symbol, amount)
#         return order

#     # Function to perform market sell
#     def market_sell(symbol, amount):
#         order = self.binance.create_market_sell_order(symbol, amount)
#         return order

#     # Function to perform limit buy
#     def limit_buy(symbol, amount, price):
#         order = self.binance.create_limit_buy_order(symbol, amount, price)
#         return order

#     # Function to perform limit sell
#     def limit_sell(symbol, amount, price):
#         order = self.binance.create_limit_sell_order(symbol, amount, price)
#         return order

#     def system_loop(self):
#         # Implement system_loop logic here
#         pass

# class KucoinSpotSystem(TradingSystem):
#     """
#         Kucoin Spot Trading System class
#         ===========================
#         This class is a specific implementation of the TradingSystem abstract base class, 
#         built to work with the Kucoin spot trading API. The KucoinSpotSystem uses the configuration,
#         member variables, and common functions defined in the TradingSystem class to manage 
#         trading operations, while adding its specific logic for interfacing with Kucoin's spot API.

#         Member Variables:
#         ===========================
#         system_id: Numerical ID of the trading system
#         system_label: String Label for the trading system 
#         symbols: Symbols of assets the user desires the system to trade 
#         strategy_weights: Dictionary of trading strategies and the weights applied to each strategy
#         congruence_level: Congruence level of trading strategies - how much should the individual strategies agree on a signal before it is made. Accepted values are low, medium, and high
#         risk_reward_ratio: Desired risk-to-reward ratio for each trade (mainly used to determine stop loss and target levels)
#         timeframe: Timeframe of data to analyze; defaults to a 15-minute chart
#         config: Configuration object containing Kucoin API credentials
#         client: Kucoin spot trading API client object
#     """

#     def __init__(self, system_id, system_label, symbols, strategy_weights, brokerage_id=3, congruence_level="medium", risk_reward_ratio=1, timeframe='15m'):
#         super().__init__(system_id=system_id, system_label=system_label, brokerage_id=brokerage_id, symbols=symbols, strategy_weights=strategy_weights, congruence_level=congruence_level, risk_reward_ratio=risk_reward_ratio, timeframe=timeframe)

#         self.brokerage_id = 3
#         # self.data_fetcher = fetch_data.FetchData()
#         # self.tech_indicators = tech_indicators.TechIndicators()

#         # Initialize Binance client
#         # self.config = self.read_config()

#         client = KucoinTrade(key=self.config.get("kucoin", "api_key"), secret=self.config.get("kucoin", "api_secret"), passphrase=self.config.get("kucoin", "api_pword"), is_sandbox=True)
        
#         self.logger.debug(f"KucoinSpotSystem initialized with ID: {self.system_id}, Label:{self.system_label}")

#     def get_balance(self):
#         results = client.get_accounts('USDT')
#         print(results) 
    
#     # Function to perform market sell
#     def market_buy(self, symbol, amount):
#         order = self.client.create_market_order(symbol, 'buy', size=str(amount))
#         self.logger.debug(f"Market Buy order sent for symbol: {symbol}")
#         return order

#     # Function to perform market sell
#     def market_sell(self, symbol, amount):
#         order = self.client.create_market_order(symbol, 'sell', size=str(amount))
#         self.logger.debug(f"Market Sell order sent for symbol: {symbol}")
#         return order

#     # Function to perform limit buy
#     def limit_buy(self, symbol, amount, price):
#         # In Kucoin Futures SDK, limit order requires size, price, and type ('buy' or 'sell')
#         order = self.client.create_limit_order(symbol, 'buy', size=str(amount), price=str(price))
#         self.logger.debug(f"Limit Buy order sent for symbol: {symbol} at price: {price}")
#         return order

#     # Function to perform limit sell
#     def limit_sell(self, symbol, amount, price):
#         order = self.client.create_limit_order(symbol, 'sell', size=str(amount), price=str(price))
#         self.logger.debug(f"Limit Sell order sent for symbol: {symbol} at price: {price}")
#         return order

#     def system_loop(self):
#         # Implement system_loop logic here
#         self.logger.debug(f"System loop started.")
#         pass

# class KucoinFuturesSystem(TradingSystem):
    
#     """
#         Kucoin Futures Trading System class
#         ===========================
#         This class is a specific implementation of the TradingSystem abstract base class, 
#         built to work with the Kucoin futures trading API. The KucoinFuturesSystem uses the configuration,
#         member variables, and common functions defined in the TradingSystem class to manage 
#         trading operations, while adding its specific logic for interfacing with Kucoin's futures API.

#         Member Variables:
#         ===========================
#         system_id: Numerical ID of the trading system
#         system_label: String Label for the trading system 
#         symbols: Symbols of assets the user desires the system to trade 
#         strategy_weights: Dictionary of trading strategies and the weights applied to each strategy
#         congruence_level: Congruence level of trading strategies - how much should the individual strategies agree on a signal before it is made. Accepted values are low, medium, and high
#         risk_reward_ratio: Desired risk-to-reward ratio for each trade (mainly used to determine stop loss and target levels)
#         timeframe: Timeframe of data to analyze; defaults to a 15-minute chart
#         config: Configuration object containing Kucoin futures API credentials
#         client: Kucoin futures trading API client object
#     """

#     def __init__(self, system_id, system_label, symbols, strategy_weights, brokerage_id=4, congruence_level="medium", risk_reward_ratio=1, timeframe='15m'):
#         super().__init__(system_id=system_id, system_label=system_label, brokerage_id=brokerage_id, symbols=symbols, strategy_weights=strategy_weights, congruence_level=congruence_level, risk_reward_ratio=risk_reward_ratio, timeframe=timeframe)
#         self.brokerage_id = 4
#         # Initialize Kucoin Futures client
#         # self.config = self.read_config()
#         self.client = KucoinFuturesTrade(key=self.config.get("kucoin_futures", "api_key"), secret=self.config.get("kucoin_futures", "api_secret"), passphrase=self.config.get("kucoin_futures", "api_pword"), is_sandbox=True)
#         self.logger.debug(f"KucoinFuturesSystem initialized with ID: {self.system_id}, Label:{self.system_label}")


#     # def analyze_asset(self, symbol):
#     #     # Implement analyze_asset logic here
#     #     self.logger.debug(f"Analysis started for symbol: {symbol}")
#     #     pass
#     def get_balance(self):
#         results = client.get_accounts('USDT')
#         print(results) 

#     # Function to perform market buy
#     def market_buy(self, symbol, amount):
#         # In Kucoin Futures SDK, market order only requires size (not amount)
#         order = self.client.create_market_order(symbol, 'buy', size=str(amount))
#         self.logger.debug(f"Market Buy order sent for symbol: {symbol} amount: {amount}")
#         return order

#     # Function to perform market sell
#     def market_sell(self, symbol, amount):
#         order = self.client.create_market_order(symbol, 'sell', size=str(amount))
#         self.logger.debug(f"Market Sell order sent for symbol: {symbol} amount: {amount}")
#         return order

#     # Function to perform limit buy
#     def limit_buy(self, symbol, amount, price):
#         # In Kucoin Futures SDK, limit order requires size, price, and type ('buy' or 'sell')
#         order = self.client.create_limit_order(symbol, 'buy', size=str(amount), price=str(price))
#         self.logger.debug(f"Limit Buy order sent for symbol: {symbol} amount: {amount} price: {price}")
#         return order

#     # Function to perform limit sell
#     def limit_sell(self, symbol, amount, price):
#         order = self.client.create_limit_order(symbol, 'sell', size=str(amount), price=str(price))
#         self.logger.debug(f"Limit Sell order sent for symbol: {symbol} amount: {amount} price: {price}")
#         return order

#     def system_loop(self):
#         # Implement system_loop logic here
#         self.logger.debug(f"System loop started.")
#         pass
