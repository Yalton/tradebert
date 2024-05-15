import modules.tradingsystem as ts

def test_trader(): 

    symbols = ['CNDA', 'AAPL', 'TSLA', 'NVDA', 'BNED', 'SPIR', 'SPCE', 'LESL', 'CHGG', 'CRM', 'PLTR', 'AMZN', 'GOOG', 'X']
    
    strategy_weights_inner = {
        'TrendFollowing': 1,
        'MomentumTrading': 1,
        'ReversalTrading': 1,
        'VolumeAnalysis': 1,
        'BreakoutTrading': 1,
        'TrendStrengthVolatility': 1,
        'VolumeFlow': 1,
        'SupportResistance': 1,
        'TrendContinuationReversal': 1,
        'MeanReversion': 1,
        'BollingerBands': 1,
        'MACD': 4,
        'SqueezeMomentum': 3,
        'CryptoLadder': 2,
    }


    strategy_weights = {symbol: strategy_weights_inner for symbol in symbols}

    AlpacaSystem = ts.AlpacaSystem(1, "AlpacaTest", symbols=symbols, strategy_weights=strategy_weights, congruence_level="medium", risk_reward_ratio=1)

    AlpacaSystem.system_loop()


if __name__ == "__main__":
    test_trader()

