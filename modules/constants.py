STOCK_MARKET_OPEN_TIME = "06:00"
STOCK_MARKET_CLOSE_TIME = "12:55"
HISTORICAL_PRICE_DATABASE = "databases/hist_data.db"


BROKERAGE_TABLE_INIT_QUERY = """
INSERT INTO Brokerage (Brokerage_ID, Brokerage_Name) VALUES
(1, 'Alpaca'),
(2, 'Binance'),
(3, 'KucoinSpot'),
(4, 'KucoinFutures')
ON DUPLICATE KEY UPDATE Brokerage_Name = VALUES(Brokerage_Name);
"""