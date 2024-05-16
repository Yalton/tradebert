
from fetch_data import DataAggregator
import pandas as pd
from tech_indicators import TechnicalIndicators
from constants import * 
import datetime
from datetime import date, timedelta
import sqlite3

# import '15min' as PredMods

class AssetAnalyzer:
    
    def __init__(self, symbol): 
        self.symbol = symbol
        self.non_TI = ["Date,Open,High,Low,Close,Adj Close,Volume"]
        self.aggregator = DataAggregator()
        self.create_table_if_not_exists()

    # def trainPredictiveModels(self): 
    #     # Initialize models
    #     self.linear_regressor = PredMods.LinearRegressor()
    #     self.rnn = PredMods.RNN()
    #     self.lstm = PredMods.LSTMModel()
    #     self.hmm = PredMods.HMM()
    #     self.arima = PredMods.ARIMAModel()

    #     # Train Models
    #     self.linear_regressor.train(self.data)
    #     self.rnn.train(self.data)
    #     self.lstm.train(self.data)
    #     self.hmm.train(self.data)
    #     self.arima.train(self.data)

    def gatherOnlineSentiment(self): 
        pass



    def calculate_greeks(S, K, T, r, sigma, option = 'call'):
        #S: spot price
        #K: strike price
        #T: time to maturity
        #r: interest rate
        #sigma: volatility of underlying asset
        #option: 'call' or 'put'
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        if option == 'call':
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)
            theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        elif option == 'put':
            delta = -norm.cdf(-d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)
            theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
        return delta, gamma, vega, theta, rho

    # def create_table_if_not_exists(self):
    #     try:
    #         self.conn = sqlite3.connect(HISTORICAL_PRICE_DATABASE)
    #         cursor = self.conn.cursor()
            
    #         create_table_query = f"""
    #             CREATE TABLE IF NOT EXISTS {self.symbol} (
    #                 Date TEXT,
    #                 Open FLOAT,
    #                 High FLOAT,
    #                 Low FLOAT,
    #                 Close FLOAT,
    #                 Adj_Close FLOAT,
    #                 Volume INT
    #             )
    #         """
            
    #         cursor.execute(create_table_query)
    #         self.conn.commit()
    #         self.conn.close()
    #     except sqlite3.Error as error:
    #         print("Failed to create SQLite table", error)

    # def fetch_most_recent_date(self):
    #     self.conn = sqlite3.connect(HISTORICAL_PRICE_DATABASE)
    #     query = f"SELECT max(Date) FROM {self.symbol}"
    #     most_recent_date = pd.read_sql_query(query, self.conn)
    #     self.conn.close()
    #     if most_recent_date.iloc[0][0] is None: 
    #         return None
    #     else:
    #         return pd.to_datetime(most_recent_date.iloc[0][0]).date()

    # def acquireData(self): 
    #     start_date = self.fetch_most_recent_date()
    #     if start_date is None:
    #         start_date = date.today() - timedelta(days=365)
    #     elif start_date < date.today():
    #         start_date = start_date + timedelta(days=1)
    #     end_date = date.today()
    #     new_data = self.aggregator.fetch_yfinance_api(self.symbol, start_date, end_date)
    #     conn = sqlite3.connect(HISTORICAL_PRICE_DATABASE)
    #     new_data.to_sql(self.symbol, conn, if_exists='append', index=False)
    #     conn.close()
        
    # # def acquireData(self): 
    # #     start_date = self.fetch_most_recent_date()
    # #     if start_date < date.today() or start_date is None:
    # #         start_date = start_date + timedelta(days=1)
    # #         end_date = date.today()
    # #         new_data = self.aggregator.fetch_yfinance_api(self.symbol, start_date, end_date)
    # #         conn = sqlite3.connect(HISTORICAL_PRICE_DATABASE)
    # #         new_data.to_sql(self.symbol, conn, if_exists='append', index=False)
    # #         conn.close()

    # def pullFromDB(self): 
    #     self.acquireData()
    #     self.conn = sqlite3.connect(HISTORICAL_PRICE_DATABASE)
    #     query = f"SELECT * FROM {self.symbol}"
    #     self.data = pd.read_sql_query(query, self.conn)
    #     self.conn.close()

    # def gatherTechIndicators(self): 

    #     # start_date = datetime.datetime.now() - datetime.timedelta(days=365)
    #     # end_date = datetime.datetime.now()
    #     # self.data = self.aggregator.fetch_yfinance_api(self.symbol, start_date, end_date)
    #     # self.data.columns = self.data.columns.str.strip()
    #     # self.data.to_csv(f"{self.symbol}.csv")
    #     self.pullFromDB()

    #     # conn = sqlite3.connect(HISTORICAL_PRICE_DATABASE)

    #     # self.data = pd.read_csv('csvs/TSLA.csv')

    #     self.TI = TechnicalIndicators(self.data)

    #     # Gather data from Technical Indicatiors 
    #     #self.data = self.data.set_index('Date')

    #     #self.data = self.data.join(self.TI.relative_strength_index(14).rename("RSI"), how='outer')
    #     self.data = self.data.join(self.TI.relative_strength_index(14).rename("RSI"), how='outer')
    #     # self.data = self.data.join(self.TI.moving_average(7).rename("7_MA"), how='outer')
    #     # self.data = self.data.join(self.TI.moving_average(25).rename("25_MA"), how='outer')
    #     # self.data = self.data.join(self.TI.moving_average(99).rename("99_MA"), how='outer')

    #     self.data['7_MA'] = self.TI.moving_average(7)
    #     self.data['25_MA'] = self.TI.moving_average(25)
    #     self.data['99_MA'] = self.TI.moving_average(99)

    #     # self.data['50_EMA'] = self.TI.exponential_moving_average(50)
    #     # self.data['ROC'] = self.TI.rate_of_change(14)
    #     # self.data['STOC_OSC'] = self.TI.stochastic_oscillator(14)
    #     # self.data['Fibonnaci_Levels'] = self.TI.fibonnaci_levels()

    #     self.data = self.data.join(self.TI.average_true_range().rename("ATR"), how='outer')
    #     self.data = self.data.join(self.TI.average_true_range().rename("OBV"), how='outer')

    #     # # Implement ichimoku_cloud
    #     # # Implement calculate_dmi
    #     # self.data['CMF'] = self.TI.calculate_cmf(14)
    #     # self.data['GANN'] = self.TI.calculate_gann(14)
    #     # self.data['VWAP'] = self.TI.calculate_vwap(14)

    #     self.add_columns_from_df()
    #     self.data.to_csv("TestMerge.csv")

    # def add_columns_from_df(self):
    #     # Map pandas dtypes to SQLite data types.
    #     dtype_mapping = {
    #         'object': 'TEXT',
    #         'int64': 'INTEGER',
    #         'float64': 'REAL',
    #         'datetime64[ns]': 'TIMESTAMP',
    #         'bool': 'INTEGER'  # SQLite does not have a separate Boolean storage class. Instead, Boolean values are stored as integers 0 (false) and 1 (true).
    #     }

    #     self.conn = sqlite3.connect(HISTORICAL_PRICE_DATABASE)
    #     cursor = self.conn.cursor()

    #     for column in self.data.columns:
    #         if column not in self.data.columns:  # Only add columns that do not already exist in the table.
    #             sqlite_dtype = dtype_mapping[str(self.data[column].dtype)]
    #             alter_table_query = f"ALTER TABLE {self.symbol} ADD COLUMN {column} {sqlite_dtype}"
    #             cursor.execute(alter_table_query)

    #     self.conn.commit()
    #     self.conn.close()
