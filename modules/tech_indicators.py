import threading
import configparser
import pandas as pd
import numpy as np
import mplfinance as mpf
#from arch import arch_model


class TechnicalIndicators(): 
    def __init__(self):
        #data = data
        pass
    
    # Define function to calculate relative strength index (RSI)
    def relative_strength_index(self, data, window):
        '''Calculate the relative strength index (RSI) of a given dataset.
        
        Args:
            data (pandas.Series): The dataset to calculate the RSI for.
            window (int): The size of the window to use for the RSI calculation.
        
        Returns:
            pandas.Series: The RSI of the given dataset.
        '''
        data = data['Close']
        delta = data.diff()
        delta = delta[1:]
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up = up.rolling(window).mean()
        roll_down = down.abs().rolling(window).mean()
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    # Define function to calculate moving average
    def moving_average(self, data, window):
        '''Calculate the moving average of a given dataset.
        
        Args:
            data (pandas.Series): The dataset to calculate the moving average for.
            window (int): The size of the window to use for the moving average calculation.
        
        Returns:
            pandas.Series: The moving average of the given dataset.
        '''
        data = data['Close']
        weights = np.repeat(1.0, window)/window
        smas = np.convolve(data, weights, 'valid')
        smas = np.concatenate((np.full(window-1, np.nan), smas))

        return smas

    # Exponential Moving Average 
    def exponential_moving_average(self, data, window):
        '''Calculate the exponential moving average (EMA) of a given dataset.

        Args:
            data (pandas.Series): The dataset to calculate the EMA for.
            window (int): The size of the window to use for the EMA calculation.

        Returns:
            pandas.Series: The EMA of the given dataset.
        '''
        data = data['Close']
        return data.ewm(span=window, adjust=False).mean()

    # Define function to calculate rate of change (ROC)
    def rate_of_change(self, data, window):
        '''Calculate the rate of change (ROC) of a given dataset.

        Args:
            data (pandas.Series): The dataset to calculate the ROC for.
            window (int): The size of the window to use for the ROC calculation.

        Returns:
            pandas.Series: The ROC of the given dataset.
        '''
        data= data['Close']
        N = data.diff(window)
        D = data.shift(window)
        roc = (N/D)*100
        roc[:window] = np.nan

        return roc



    # Define function to calculate stochastic oscillator
    def stochastic_oscillator(self, data, n):
        '''Calculate the stochastic oscillator of a given dataset.
        
        Args:
            high (pandas.Series): The high values of the dataset.
            low (pandas.Series): The low values of the dataset.
            close (pandas.Series): The close values of the dataset.
            n (int): The size of the window to use for the stochastic oscillator calculation.
        
        Returns:
            pandas.Series: The stochastic oscillator of the given dataset.
        '''
        high = data['High']
        low = data['Low']
        close = data['Close']
        lowest_low = low.rolling(window=n).min()
        highest_high = high.rolling(window=n).max()
        k = 100 * (close - lowest_low)/(highest_high - lowest_low)
        return k


    def fibonnaci_levels(self, data): 
        close = data['Close']
        high, low = close.max(), close.min()
        diff = high - low
        levels = [high, high - diff*0.236, high - diff*0.382, high - diff*0.5, high - diff*0.618, low]

        return levels
    
    # On-Balance Volume (OBV): This is a technical trading momentum indicator that uses volume flow to predict changes in stock price.
    def on_balance_volume(self, data): 
        close = data['Close']
        volume = data['Volume']  # Add this line
        obv_values = [0]
        for i in range(1, len(close)):
            if close[i] > close[i-1]:  
                obv_values.append(obv_values[-1] + volume[i]) 
            elif close[i] < close[i-1]:  
                obv_values.append(obv_values[-1] - volume[i])  
            else: 
                obv_values.append(obv_values[-1]) 

        return obv_values


    # Ichimoku Cloud: This is a collection of technical indicators that show levels of support and resistance, direction of trend, and momentum. 
    # The "cloud" is a shaded area on the chart showing where support and resistance levels are.
    def ichimoku_cloud(self, data): 
        # Ichimoku Cloud
        high_prices = data['High']
        low_prices = data['Low']
        close_prices = data['Close']

        nine_period_high = high_prices.rolling(window=9).max()
        nine_period_low = low_prices.rolling(window=9).min()
        tenkan_sen = (nine_period_high + nine_period_low) /2

        # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
        period26_high = high_prices.rolling(window=26).max()
        period26_low = low_prices.rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2

        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
        period52_high = high_prices.rolling(window=52).max()
        period52_low = low_prices.rolling(window=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

        # The most current closing price plotted 22 time periods behind (optional)
        chikou_span = close_prices.shift(-22) # 22 according to investopedia

        return kijun_sen, senkou_span_a, senkou_span_b, chikou_span

    # Average True Range (ATR): ATR is a technical analysis volatility indicator originally developed by J. Welles Wilder, Jr. for commodities. 
    # The indicator does not provide an indication of price trend; it merely measures the degree of price volatility.
    def average_true_range(self, data, window): 
        def calculate_true_range(high, low, close):
            return np.maximum(np.maximum(high - low, abs(high - close.shift())), abs(low - close.shift()))

        high_prices = data['High']
        low_prices = data['Low']
        close_prices = data['Close']

        true_range = calculate_true_range(high_prices, low_prices, close_prices)
        atr = true_range.rolling(window=window).mean()  # Typically, 14 periods are used
        return atr

    # Directional Movement Index (DMI): This is an indicator that identifies whether a security is trending by comparing successive highs and lows.
    def calculate_dmi(self, data, period):
        data['UpMove'] = data['High'] - data['High'].shift(1)
        data['DownMove'] = data['Low'].shift(1) - data['Low']
        data['Zero'] = 0

        data['PlusDM'] = np.where((data['UpMove'] > data['DownMove']) & (data['UpMove'] > data['Zero']), data['UpMove'], 0)
        data['MinusDM'] = np.where((data['DownMove'] > data['UpMove']) & (data['DownMove'] > data['Zero']), data['DownMove'], 0)

        TR = calculate_true_range(data['High'], data['Low'], data['Close'])
        data['TR'] = TR
        data['ATR'] = data['TR'].rolling(window=period).sum()

        data['PlusDI'] = (data['PlusDM'].rolling(window=period).sum() / data['ATR']) * 100
        data['MinusDI'] = (data['MinusDM'].rolling(window=period).sum() / data['ATR']) * 100

        data['ADX'] = (abs(data['PlusDI'] - data['MinusDI']) / (data['PlusDI'] + data['MinusDI'])) * 100
        data['ADX'] = data['ADX'].rolling(window=period).mean()

        return data

    # Chaikin Money Flow (CMF): This is an oscillator that measures buying and selling pressure over a set period of time.
    def chaikin_money_flow(self, data, period):
        clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
        cmf = ((clv * data['Volume']).rolling(window=period).sum()) / data['Volume'].rolling(window=period).sum()
        return cmf

    # Gann Angles: This is a method for predicting price movements through the relation of geometric angles in charts depicting time and price.
    def calculate_gann(self, data):
        timedelta = data.index - data.index[0]
        days = timedelta.days
        return (data['High'] - data['Low']) / days

    # Volume-Weighted Average Price (VWAP): VWAP is the average price a security has traded at throughout the day, 
    # based on both volume and price.
    def calculate_vwap(self, data):
        return data['Close'].cumsum() / data['Volume'].cumsum()

    # Pivot Points: These are used to identify intraday support, resistance and target levels. 
    # The pivot point itself is simply the average of the high, low and closing prices from the previous trading day.
    def calculate_pivot_points(self, data):
        high = data['High']
        low = data['Low']
        close = data['Close']

        pivot_point = (high + low + close) / 3

        s1 = (pivot_point * 2) - high
        s2 = pivot_point - (high - low)

        r1 = (pivot_point * 2) - low
        r2 = pivot_point + (high - low)

        return pivot_point, s1, s2, r1, r2



    # Calculate the moving average convergence divergence of a given data series
    def macd(self, data):
        close = data['Close']
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal



    
    # Function to plot.
    def plot_data(self, data):
        data = data[-90:]
        data['12-SMA'] = data.Close.rolling(window=12).mean()
        data['26-SMA'] = data.Close.rolling(window=26).mean()
        mpf.plot(data,type='line',mav=(12,26),figsize=(15,7), title='Technical Analysis',
                ylabel='Price (USD)')



    # # Moving Average Crossover (MAC) trading strategy 
    # def mac_strategy(self):
    #     short_window = 50
    #     long_window = 200

    #     signals = pd.DataFrame(index=data.index)
    #     signals['signal'] = 0.0

    #     signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    #     signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

    #     signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
    #     signals['positions'] = signals['signal'].diff()

    #     if signals['positions'].iloc[-1] == 1.0:
    #         mac_signal = "BUY"
    #     elif signals['positions'].iloc[-1] == -1.0:
    #         mac_signal = "SELL"
    #     else:
    #         mac_signal = "HOLD"
            
    #     return mac_signal