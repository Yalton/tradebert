from  tradingsystem import TradingSystem
from  tradingsystem import AlpacaPaperSocket

class PortfolioManagementSystem(TradingSystem):

    def __init__(self):
        super().__init__(AlpacaPaperSocket(), 'IBM', 604800, 1, 'AI_PM')

    def analyze_asset(self):
        pass

    def market_buy(self):
        pass

    def market_sell(self):
        pass

    def limit_buy(self):
        pass

    def limit_sell(self): 
        pass

    def system_loop(self):
        pass