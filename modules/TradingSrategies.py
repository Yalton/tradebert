from modules import tech_indicators as tech_ind
#import asset_analysis as AA
from .fetch_data import DataAggregator
import ta
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
tqdm.pandas()


class BaseStrategy:
    def __init__(self, data, weight=1):
        self.data = data
        # self.start_date = start_date
        # self.end_date = end_date
        self.weight = weight

        # aggregator = DataAggregator()
        # start_date = datetime.datetime.now() - datetime.timedelta(days=365)
        # end_date = datetime.datetime.now()
        # with tqdm(total=1, disable=True):
        #     self.data = aggregator.fetch_yfinance_api(self.data, start_date, end_date)
        # self.data.columns = self.data.columns.str.strip()
        self.TI = tech_ind.TechnicalIndicators()

    def get_dataframe(self): 
        return self.data

    def generate_signal(self):
        raise NotImplementedError("This method should be overridden by child class")


class TrendFollowingStrategy(BaseStrategy):
    def __init__(self, weight, data, ma_window=50):
        super().__init__(weight=weight, data=data)
        self.ma_window = ma_window

    def generate_signal(self):
        self.data['MA'] = self.data['Close'].rolling(window=self.ma_window).mean()
        signal = np.where(self.data['Close'] > self.data['MA'], 1, -1)
        return (self.weight * signal)

# Momentum Trading Strategy
class MomentumTradingStrategy(BaseStrategy):
    def __init__(self, weight, data, roc_window=14):
        super().__init__(weight=weight, data=data)
        self.roc_window = roc_window

    def generate_signal(self):
        self.data['ROC'] = ((self.data['Close'] - self.data['Close'].shift(self.roc_window)) / self.data['Close'].shift(self.roc_window)) * 100
        signal = np.where(self.data['ROC'] > 0, 1, -1)
        return (self.weight * signal)


class TrendFollowingStrategy(BaseStrategy):
    def __init__(self, weight, data, ma_window=50):
        super().__init__(weight=weight, data=data)
        self.ma_window = ma_window

    def generate_signal(self):
        self.data['MA'] = self.data['Close'].rolling(window=self.ma_window).mean()
        signal = np.where(self.data['Close'] > self.data['MA'], 1, -1)
        return (self.weight * signal)


class ReversalTradingStrategy(BaseStrategy):
    def __init__(self, weight, data, rsi_window=14, overbought=70, oversold=30):
        super().__init__(weight=weight, data=data)
        self.rsi_window = rsi_window
        self.overbought = overbought
        self.oversold = oversold

    def generate_signal(self):
        self.data['RSI'] = self.TI.relative_strength_index(self.data, self.rsi_window)
        return  (self.weight * np.where(self.data['RSI'] > self.overbought, -1,  np.where(self.data['RSI'] < self.oversold, 1, 0)))


class VolumeAnalysisStrategy(BaseStrategy):
    def __init__(self, weight, data):
        super().__init__(weight=weight, data=data)

    def generate_signal(self):
        self.data['OBV'] = self.TI.on_balance_volume(self.data)
        return  (self.weight * np.where(self.data['OBV'] > self.data['OBV'].shift(1), 1, -1))


class BreakoutTradingStrategy(BaseStrategy):
    def __init__(self, weight, data, window=14):
        super().__init__(weight=weight, data=data)
        self.window = window

    def generate_signal(self):
        self.data['High'] = self.data['Close'].rolling(window=self.window).max()
        self.data['Low'] = self.data['Close'].rolling(window=self.window).min()
        Buy_Sig = np.where(self.data['Close'] > self.data['High'].shift(1), 1, 0)
        Sell_Sig = np.where(self.data['Close'] < self.data['Low'].shift(1), -1, 0)
        return (self.weight * (Buy_Sig + Sell_Sig))

class TrendStrengthVolatilityStrategy(BaseStrategy):
    def __init__(self, weight, data, window=14):
        super().__init__(weight=weight, data=data)
        self.window = window

    def generate_signal(self):
        self.data['ATR'] = self.TI.average_true_range(self.data, self.window)
        self.data['MA'] = self.data['Close'].rolling(window=self.window).mean()
        return  (self.weight * np.where((self.data['Close'] > self.data['MA']) & (self.data['ATR'] > self.data['ATR'].shift(1)), 1, 
                        np.where((self.data['Close'] < self.data['MA']) & (self.data['ATR'] < self.data['ATR'].shift(1)), -1, 0)))

class VolumeFlowStrategy(BaseStrategy):
    def __init__(self, weight, data):
        super().__init__(weight=weight, data=data)

    def generate_signal(self):
        self.data['CMF'] = self.TI.chaikin_money_flow(self.data, 20) # Assuming self.TI.chaikin_money_flow is your function to calculate CMF
        return  (self.weight * np.where(self.data['CMF'] > 0, 1, np.where(self.data['CMF'] < 0, -1, 0)))

class SupportResistanceStrategy(BaseStrategy):
    def __init__(self, weight, data, window=14):
        super().__init__(weight=weight, data=data)
        self.window = window

    def generate_signal(self):
        self.data['Resistance'] = self.data['Close'].rolling(window=self.window).max()
        self.data['Support'] = self.data['Close'].rolling(window=self.window).min()
        return  (self.weight * np.where(self.data['Close'] < (self.data['Support'].shift(1) * 1.03), 1, 
                        np.where(self.data['Close'] > (self.data['Resistance'].shift(1) * 0.97), -1, 0)))

class TrendContinuationReversalStrategy(BaseStrategy):
    def __init__(self, weight, data, window=14):
        super().__init__(weight=weight, data=data)
        self.window = window

    def generate_signal(self):
        self.data['MA'] = self.data['Close'].rolling(window=self.window).mean()
        self.data['Momentum'] = self.data['Close'] - self.data['MA']
        return  (self.weight * np.where((self.data['Momentum'] > 0) & (self.data['Momentum'] > self.data['Momentum'].shift(1)), 1, np.where((self.data['Momentum'] < 0) & (self.data['Momentum'] < self.data['Momentum'].shift(1)), -1, 0)))


class MeanReversionStrategy(BaseStrategy):
    def __init__(self, weight, data, window=14):
        super().__init__(weight=weight, data=data)
        self.window = window

    def generate_signal(self):
        self.data['MA'] = self.data['Close'].rolling(window=self.window).mean()
        self.data['Momentum'] = self.data['Close'] - self.data['MA']
        return  (self.weight * np.where((self.data['Momentum'] > 0) & (self.data['Momentum'] > self.data['Momentum'].shift(1)), 1, np.where((self.data['Momentum'] < 0) & (self.data['Momentum'] < self.data['Momentum'].shift(1)), -1, 0)))


class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, weight, data, window=20, num_std=2):
        super().__init__(weight=weight, data=data)
        self.window = window
        self.num_std = num_std

    def generate_signal(self):
        self.data['SMA'] = self.data['Close'].rolling(self.window).mean()
        self.data['stddev'] = self.data['Close'].rolling(self.window).std()
        self.data['BBUP'] = self.data['SMA'] + self.num_std * self.data['stddev']
        self.data['BBLOW'] = self.data['SMA'] - self.num_std * self.data['stddev']
        
        return (self.weight * np.where(self.data['Close'] > self.data['BBUP'], -1, 
                        np.where(self.data['Close'] < self.data['BBLOW'], 1, 0)))


class MACDStrategy(BaseStrategy):
    def __init__(self, weight, data, n_fast=12, n_slow=26, n_sign=9):
        super().__init__(weight=weight, data=data)
        self.n_fast = n_fast
        self.n_slow = n_slow
        self.n_sign = n_sign

    def generate_signal(self):
        self.data['ema200'] = ta.trend.ema_indicator(self.data['Close'], window=200)
        self.data['macd_line'] = ta.trend.macd_diff(self.data['Close'], self.n_slow, self.n_fast, self.n_sign)
        
        # Generate signals
        return (self.weight * np.where((self.data['macd_line'] > 0) & (self.data['Close'] > self.data['ema200']), 1, 
                        np.where((self.data['macd_line'] < 0) & (self.data['Close'] < self.data['ema200']), -1, 0)))

class SqueezeMomentumStrategy(BaseStrategy):
    def __init__(self, weight, data, length=20, mult=2.0, lengthKC=20, multKC=1.5, useTrueRange=True):
        super().__init__(weight=weight, data=data)
        self.length = length
        self.mult = mult
        self.lengthKC = lengthKC
        self.multKC = multKC
        self.useTrueRange = useTrueRange

    def generate_signal(self):
        # Calculate BB
        self.data['basis'] = self.data['Close'].rolling(window=self.length).mean()
        self.data['dev'] = self.multKC * self.data['Close'].rolling(window=self.length).std()
        self.data['upperBB'] = self.data['basis'] + self.data['dev']
        self.data['lowerBB'] = self.data['basis'] - self.data['dev']

        # Calculate KC
        self.data['ma'] = self.data['Close'].rolling(window=self.lengthKC).mean()
        self.data['range'] = self.data['High'] - self.data['Low'] if not self.useTrueRange else np.maximum(np.maximum(self.data['High'] - self.data['Low'], abs(self.data['High'] - self.data['Close'].shift())), abs(self.data['Low'] - self.data['Close'].shift()))
        self.data['range_ma'] = self.data['range'].rolling(window=self.lengthKC).mean()
        self.data['upperKC'] = self.data['ma'] + self.data['range_ma'] * self.multKC
        self.data['lowerKC'] = self.data['ma'] - self.data['range_ma'] * self.multKC

        # Calculate Squeeze
        self.data['val'] = np.nan
        for i in range(self.lengthKC - 1, len(self.data)):
            y = self.data['Close'].iloc[i - self.lengthKC + 1 : i + 1] - ((self.data['High'].iloc[i - self.lengthKC + 1 : i + 1] + self.data['Low'].iloc[i - self.lengthKC + 1 : i + 1]) / 2)
            x = np.array(range(self.lengthKC)).reshape(-1,1)
            model = LinearRegression()
            model.fit(x, y)
            self.data.loc[self.data.index[i], 'val'] = model.coef_[0]
            #self.data['val'].iloc[i] = model.coef_[0]

        # Trading Signals
        return (self.weight * np.where(self.data['val'] > 0, 1, 
                        np.where(self.data['val'] < 0, -1, 0)))

class CryptoLadderStrategy(BaseStrategy):
    def __init__(self, weight, data, per=100, mult=3.0):
        super().__init__(weight=weight, data=data)
        self.per = per
        self.mult = mult

    def generate_signal(self):
        # calculate the smoothed average range
        self.data['range'] = abs(self.data['Close'] - self.data['Close'].shift())
        self.data['av_range'] = ta.trend.ema_indicator(self.data['range'], self.per)
        self.data['smooth_range'] = ta.trend.ema_indicator(self.data['av_range'], self.per * 2 - 1) * self.mult

        # range filter
        self.data['filt'] = self.data['Close']
        

        for i in range(1, len(self.data)):
            close_loc = self.data.columns.get_loc('Close')
            filt_loc = self.data.columns.get_loc('filt')
            smooth_range_loc = self.data.columns.get_loc('smooth_range')
            if self.data.iat[i, close_loc] > self.data.iat[i - 1, filt_loc]:
                self.data.iat[i, filt_loc] = max(self.data.iat[i, close_loc] - self.data.iat[i, smooth_range_loc], self.data.iat[i - 1, filt_loc])
            else:
                self.data.iat[i, filt_loc] = min(self.data.iat[i, close_loc] + self.data.iat[i, smooth_range_loc], self.data.iat[i - 1, filt_loc])
                
        # for i in range(1, len(self.data)):
        #     if self.data.iloc[i][self.data.columns.get_loc('Close')] > self.data.iloc[i - 1][self.data.columns.get_loc('filt')]:
        #         self.data.iloc[i][self.data.columns.get_loc('filt')] = max(self.data.iloc[i][self.data.columns.get_loc('Close')] - self.data.iloc[i][self.data.columns.get_loc('smooth_range')], self.data.iloc[i - 1][self.data.columns.get_loc('filt')])
        #     else:
        #         self.data.iloc[i][self.data.columns.get_loc('filt')] = min(self.data.iloc[i][self.data.columns.get_loc('Close')] + self.data.iloc[i][self.data.columns.get_loc('smooth_range')], self.data.iloc[i - 1][self.data.columns.get_loc('filt')])

        # filter direction
        self.data['upward'] = (self.data['filt'] > self.data['filt'].shift()).cumsum()
        self.data['downward'] = (self.data['filt'] < self.data['filt'].shift()).cumsum()

        # target bands
        self.data['hband'] = self.data['filt'] + self.data['smooth_range']
        self.data['lband'] = self.data['filt'] - self.data['smooth_range']

        # break outs
        self.data['longCond'] = ((self.data['Close'] > self.data['filt']) & (self.data['Close'] > self.data['Close'].shift()) & (self.data['upward'] > 0)) | ((self.data['Close'] > self.data['filt']) & (self.data['Close'] < self.data['Close'].shift()) & (self.data['upward'] > 0))
        self.data['shortCond'] = ((self.data['Close'] < self.data['filt']) & (self.data['Close'] < self.data['Close'].shift()) & (self.data['downward'] > 0)) | ((self.data['Close'] < self.data['filt']) & (self.data['Close'] > self.data['Close'].shift()) & (self.data['downward'] > 0))
        self.data['CondIni'] = np.where(self.data['longCond'], 1, np.where(self.data['shortCond'], -1, np.nan))
        self.data['CondIni'].fillna(method='ffill', inplace=True)
        self.data['longCondition'] = self.data['longCond'] & (self.data['CondIni'].shift() == -1)
        self.data['shortCondition'] = self.data['shortCond'] & (self.data['CondIni'].shift() == 1)

        # Return signals
        return (self.weight * np.where(self.data['longCondition'], 1, np.where(self.data['shortCondition'], -1, 0)))

