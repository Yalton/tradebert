from modules import tradingsystem as ts

def interface(): 
    # Initialize the strategy weights
    strategy_weights_inner = {'TrendFollowing': 1, 'MomentumTrading': 1, 'ReversalTrading': 1, 'VolumeAnalysis': 1, 
                            'BreakoutTrading': 1, 'TrendStrengthVolatility': 1, 'VolumeFlow': 1, 'SupportResistance': 1, 
                            'TrendContinuationReversal': 1, 'MeanReversion': 1, 'BollingerBands': 1, 'MACD': 2, 
                            'SqueezeMomentum': 2, 'CryptoLadder': 2}

    # List of all symbols
    symbols = ['CNDA', 'AAPL', 'TSLA', 'NVDA', 'BNED', 'SPIR', 'SPCE', 'LESL', 'CHGG', 'CRM', 'PLTR', 'AMZN', 'GOOG', 'X']

    # Create the dictionary for each symbol
    strategy_weights = {symbol: strategy_weights_inner for symbol in symbols}

    AlpacaSystem = ts.AlpacaSystem(1, "AlpacaTest", symbols=symbols, strategy_weights=strategy_weights, congruence_level="low", risk_reward_ratio=1)


if __name__ == "__main__": 
    interface()


