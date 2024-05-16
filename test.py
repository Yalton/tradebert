import datetime

import configparser

# import predictive_models as pm
import modules.tradingsystem as ts
from modules.fetch_data import DataAggregator
from modules.tech_indicators import TechnicalIndicators
import modules.TradingSrategies as TS

# from asset_analysis import AssetAnalyzer

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from modules.constants import *
import sqlite3


def generate_ensemble_signal(strategy_list):
    total_signal = 0
    for strategy in strategy_list:
        total_signal += strategy.generate_signal()

    return total_signal

# Read from Config file and import api keys
def read_config():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config


def test_data_aggregator():
    config = read_config()
    aggregator = DataAggregator(config.get("alphavantage", "api_key"))


    # Test fetch_crypto_news
    # print("Testing fetch_crypto_news:")
    # aggregator.fetch_crypto_news()

    # Test extract_financial_news
    # print("\nTesting extract_financial_news:")
    # articles = aggregator.extract_financial_news()
    # if articles:
    #     print(f"Found {len(articles)} articles from MarketWatch.")
    # else:
    #     print("No articles found.")

    # print("\nTesting fetch_asset_prices:")
    # # Test fetch_asset_prices
    # symbols = ['BTC', 'ETH']  # Add your symbols here
    # directory = "./csvs"  # Add your directory here
    # aggregator.fetch_asset_prices(symbols, directory, False)
    # print("Exception thrown")

    # Test fetch_yfinance_api
    print("\nTesting data fetcher:")
    start_date = datetime.datetime.now() - datetime.timedelta(days=50)

    end_date = datetime.datetime.now()
    data = aggregator.fetch_yfinance_api("AAPL", start_date, end_date)
    print(data)
    data.to_csv("csvs/AAPL2.csv")


def test_tech_indicators():

    # Initialize TechnicalIndicators class

    # Initialize Aggregator class
    aggregator = DataAggregator()

    # Fetch historical data
    # start_date = datetime.datetime.now() - datetime.timedelta(days=365)
    # end_date = datetime.datetime.now()
    # data = aggregator.fetch_yfinance_api('TSLA', start_date, end_date)

    data = pd.read_csv("csvs/TSLA.csv")
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date", inplace=True)
    TI = TechnicalIndicators(data)

    # data = pd.read_csv('csvs/AAPL.csv')
    # Calculate moving average

    RSI = TI.relative_strength_index(14).rename("RSI")
    FIFTY_MA = TI.moving_average(50)
    FIFTY_EMA = TI.exponential_moving_average(50)
    ROC = TI.rate_of_change(14).rename("ROC")
    STOC_OSC = TI.stochastic_oscillator(14).rename("STOC_OSC")
    Fibonnaci_Levels = TI.fibonnaci_levels()
    ATR = TI.average_true_range().rename("ATR")
    OBV = TI.on_balance_volume()
    CMF = TI.calculate_cmf(14).rename("CMF")
    GANN = TI.calculate_gann().rename("GANN")
    VWAP = TI.calculate_vwap().rename("VWAP")

    # ATR.drop(['Date'], axis=1)
    ATR.to_csv("FIFTY_MA.csv")

    print("RSI:")
    print(RSI)

    print("FIFTY_MA:")
    print(FIFTY_MA)

    print("FIFTY_EMA:")
    print(FIFTY_EMA)

    print("ROC:")
    print(ROC)

    print("STOC_OSC:")
    print(STOC_OSC)

    print("Fibonnaci_Levels:")
    print(Fibonnaci_Levels)

    print("ATR:")
    print(ATR)

    print("OBV:")
    print(OBV)

    print("CMF:")
    print(CMF)

    print("GANN:")
    print(GANN)

    print("VWAP:")
    print(VWAP)

    # Plot data
    # ti.plot_data(data)


# def test_models():
#     testLinRegress = False
#     testRNN = False
#     testLTSM = False
#     testHMM = True
#     testArima = False

#     # Use pandas to load your data.
#     data = pd.read_csv('csvs/TSLA.csv')
#     symbol = 'TSLA'

#     # Initialize TechnicalIndicators class
#     TI = TechnicalIndicators(data)

#     data['MA_50'] = TI.moving_average(window=50)
#     data['ROC'] = TI.rate_of_change(window=10)


#     # Test models on some data
#     X_test = data.iloc[-10:]  # last 10 rows

#     avg_test = []
#     if(testLinRegress):
#         linear_regressor = pm.LinearRegressor(symbol)
#         print("Training LinearRegressor...")
#         linear_regressor.train(data)
#         X_test = data.iloc[-1:]  # last 10 rows
#         print("LinearRegressor Predictions:", linear_regressor.predict(X_test))
#         print("Actual ", X_test['Close'])
#         avg_test.append(linear_regressor.predict(X_test))


#     if(testRNN):
#         data['MA_50'] = TI.moving_average(data['Close'], window=50)
#         data['EMA'] = TI.exponential_moving_average(data['Close'], window=50)
#         data['ROC'] = TI.rate_of_change(data['Close'], window=10)
#         X_test = data.iloc[-1:]  # last 10 rows
#         rnn = pm.RNN(symbol)
#         print("Training RNN...")
#         rnn.train(data)
#         print("RNN Predictions:", rnn.predict(X_test))
#         print("Actual ", X_test['Close'])
#         avg_test.append(rnn.predict(X_test))


#         #rnn.predict(X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))))

#     if(testLTSM):
#         data['MA_50'] = TI.moving_average(data['Close'], window=50)
#         data['EMA'] = TI.exponential_moving_average(data['Close'], window=50)
#         data['ROC'] = TI.rate_of_change(data['Close'], window=10)
#         X_test = data.iloc[-1:]  # last 10 rows
#         lstm = pm.LSTMModel(symbol)
#         print("Training LSTM...")
#         lstm.train(data)
#         print("LSTM Predictions:", lstm.predict(X_test))
#         X_test = data.iloc[-1:]  # last 10 rows
#         print("Actual ", X_test['Close'])
#         avg_test.append(lstm.predict(X_test))

#     if(testHMM):
#         hmm = pm.HMM(symbol)
#         print("Training HMM...")
#         hmm.train(data)
#         print("HMM Predictions:", hmm.predict(10))
#         print("HMM details:")
#         hmm.print_details()

#     if(testArima):
#         X_test = data.iloc[-1:]  # last 10 rows
#         arima = pm.ARIMAModel(symbol)
#         print("Training arima...")
#         arima.train(data)
#         Y_pred = arima.predict(X_test)
#         print("arima Predictions:", Y_pred)
#         print("arima Forecast:", arima.forecast(10))
#         #forecast = arima.forecast(10)
#         # r2 = r2_score(X_test['Close'], Y_pred)
#         # print('R^2 Score:', r2)
#         #arima.print_details()


#     # avg = np.sum(avg_test) / 3
#     # print("Average", avg)
#     # print("Actual ", X_test['Close'])


def test_AssetAnalyzer():
    # Instantiate the class
    analyzer = AssetAnalyzer("AAPL")  # Use 'AAPL' as the symbol

    # Gather technical indicators
    analyzer.gatherTechIndicators()
    data = analyzer.data

    # Check the calculated technical indicators
    print("Moving Average 7:\n", data["7_MA"])
    print("Moving Average 25:\n", data["25_MA"])
    print("Moving Average 99:\n", data["99_MA"])
    print("Relative Strength Index:\n", data["RSI"])

    print("ATR:\n", data["ATR"])
    print("OBV:\n", data["OBV"])

    # print("Stochastic Oscillator:\n", data['STOC_OSC'])
    # print("Fibonnaci_Levels:\n", data['Fibonnaci_Levels'])
    # print("CMF:\n", data['CMF'])
    # print("GANN:\n", data['GANN'])
    # print("VWAP:\n", data['VWAP'])

    # print("MACD:\n", data)
    # print("MACD Signal:\n", data)

    # analyzer.trainPredictiveModels()


# def test_trader():
#     AlpacaSystem = ts.AlpacaSystem(1, "TestTrader")
#     #AlpacaSystem.test()
#     AlpacaSystem.market_buy("TLSA", 1)


def test_trader():
    # AlpacaSystem = ts.AlpacaSystem(1, "AlpacaTest")
    # BinanceSystem = ts.BinanceSystem(2, "BinanceTest")
    # KucoinSpotSystem = ts.KucoinSpotSystem(3, "KucoinTest")
    # KucoinFuturesSystem = ts.KucoinFuturesSystem(4, "KucoinFuturesTest")

    symbols = [
        "CNDA",
        "AAPL",
        "TSLA",
        "NVDA",
        "BNED",
        "SPIR",
        "SPCE",
        "LESL",
        "CHGG",
        "CRM",
        "PLTR",
        "AMZN",
        "GOOG",
        "X",
    ]

    strategy_weights_inner = {
        "TrendFollowing": 1,
        "MomentumTrading": 1,
        "ReversalTrading": 1,
        "VolumeAnalysis": 1,
        "BreakoutTrading": 1,
        "TrendStrengthVolatility": 1,
        "VolumeFlow": 1,
        "SupportResistance": 1,
        "TrendContinuationReversal": 1,
        "MeanReversion": 1,
        "BollingerBands": 1,
        "MACD": 2,
        "SqueezeMomentum": 2,
        "CryptoLadder": 2,
    }

    strategy_weights = {symbol: strategy_weights_inner for symbol in symbols}

    AlpacaSystem = ts.AlpacaSystem(
        1,
        "AlpacaTest",
        symbols=symbols,
        strategy_weights=strategy_weights,
        congruence_level="low",
        risk_reward_ratio=1,
    )

    AlpacaSystem.system_loop()
    # AlpacaSystem.market_buy("TLSA", 1)

    # # print(signals)

    # AlpacaSystem = ts.AlpacaSystem(1, "AlpacaTest", symbols=symbols, strategy_weights=strategy_weights, congruence_level="low", risk_reward_ratio=2)

    # buying_power = AlpacaSystem.get_balance()

    # print(f"Buying Power {buying_power}")

    # print(f"Current Price {AlpacaSystem.get_current_price('AAPL')}")

    # print(f"Trade Amount {AlpacaSystem.calculate_trade_amount('Buy')}")

    # print(f"Take Profit Price {AlpacaSystem.calculate_take_profit_price('AAPL')}")

    # print(f"Stop Price Amount {AlpacaSystem.calculate_stop_price('AAPL')}")


def test_Strategies():

    # print(signals)
    symbols = [
        "CNDA",
        "AAPL",
        "TSLA",
        "NVDA",
        "BNED",
        "SPIR",
        "SPCE",
        "LESL",
        "CHGG",
        "CRM",
        "PLTR",
        "AMZN",
        "GOOG",
        "X",
    ]
    strategy_weights_inner = {
        "TrendFollowing": 1,
        "MomentumTrading": 1,
        "ReversalTrading": 1,
        "VolumeAnalysis": 1,
        "BreakoutTrading": 1,
        "TrendStrengthVolatility": 1,
        "VolumeFlow": 1,
        "SupportResistance": 1,
        "TrendContinuationReversal": 1,
        "MeanReversion": 1,
        "BollingerBands": 3,
        "MACD": 5,
        "SqueezeMomentum": 3,
        "CryptoLadder": 1,
    }

    strategy_weights = {symbol: strategy_weights_inner for symbol in symbols}

    system = ts.AlpacaSystem(
        1,
        "AlpacaTest",
        symbols=symbols,
        strategy_weights=strategy_weights,
        congruence_level="medium",
    )
    return system.analyze_assets()


def test_order():
    symbols = ["CNDA", "AAPL", "TSLA"]
    strategy_weights_inner = {
        "TrendFollowing": 1,
        "MomentumTrading": 1,
        "ReversalTrading": 1,
        "VolumeAnalysis": 1,
        "BreakoutTrading": 1,
        "TrendStrengthVolatility": 1,
        "VolumeFlow": 1,
        "SupportResistance": 1,
        "TrendContinuationReversal": 1,
        "MeanReversion": 1,
        "BollingerBands": 1,
        "MACD": 2,
        "SqueezeMomentum": 2,
        "CryptoLadder": 2,
    }

    symbol = "TSLA"

    strategy_weights = {symbol: strategy_weights_inner for symbol in symbols}

    system = ts.AlpacaSystem(
        1,
        "AlpacaTest",
        symbols=symbols,
        strategy_weights=strategy_weights,
        congruence_level="medium",
    )

    system.bracket_order(symbol, 1, system.get_current_price(symbol))


if __name__ == "__main__":

    test_data_aggregator()
    print("Strategy, Results. ", test_Strategies())
    # test_trader()
    # test_order()
