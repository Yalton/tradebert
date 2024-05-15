USE tradeBert;

-- Brokerage Table
CREATE TABLE IF NOT EXISTS Brokerage (
    Brokerage_ID INT PRIMARY KEY,
    Brokerage_Name VARCHAR(255)
);

-- SystemTable
CREATE TABLE IF NOT EXISTS SystemTable (
    System_ID INT PRIMARY KEY,
    System_Name VARCHAR(255),
    Brokerage_ID INT,
    FOREIGN KEY (Brokerage_ID) REFERENCES Brokerage(Brokerage_ID) ON DELETE CASCADE
);

-- Asset Table
CREATE TABLE IF NOT EXISTS Asset (
    Asset_ID INT AUTO_INCREMENT PRIMARY KEY,
    System_ID INT,
    Symbol VARCHAR(255),
    Tradable BOOLEAN,
    Asset_Class VARCHAR(255),
    -- FOREIGN KEY (Brokerage_ID) REFERENCES Brokerage(Brokerage_ID) ON DELETE CASCADE
    UNIQUE (System_ID, Symbol),
    FOREIGN KEY (System_ID) REFERENCES SystemTable(System_ID) ON DELETE CASCADE
);

-- Signal Table
CREATE TABLE IF NOT EXISTS SignalTable (
    Signal_ID INT AUTO_INCREMENT PRIMARY KEY,
    -- System_ID INT,
    Asset_ID INT,
    Trade_Signal VARCHAR(255),
    Congruence_Signal INT,
    Congruence_Level VARCHAR(255),
    TrendFollowing INT,
    MomentumTrading INT,
    ReversalTrading INT,
    VolumeAnalysis INT,
    BreakoutTrading INT,
    TrendStrengthVolatility INT,
    VolumeFlow INT,
    SupportResistance INT,
    TrendContinuationReversal INT,
    MeanReversion INT,
    BollingerBands INT,
    MACD INT,
    SqueezeMomentum INT,
    CryptoLadder INT, 
    Timestamp TIMESTAMP, 
    UNIQUE (Asset_ID),
    -- FOREIGN KEY (System_ID) REFERENCES SystemTable(System_ID) ON DELETE CASCADE,
    FOREIGN KEY (Asset_ID) REFERENCES Asset(Asset_ID) ON DELETE CASCADE
);

-- StrategyWeights Table
CREATE TABLE IF NOT EXISTS StrategyWeights (
    Strategy_Weight_ID INT AUTO_INCREMENT PRIMARY KEY,
    -- System_ID INT,
    Asset_ID INT,
    -- FOREIGN KEY (System_ID) REFERENCES SystemTable(System_ID) ON DELETE CASCADE,
    TrendFollowing INT,
    MomentumTrading INT,
    ReversalTrading INT,
    VolumeAnalysis INT,
    BreakoutTrading INT,
    TrendStrengthVolatility INT,
    VolumeFlow INT,
    SupportResistance INT,
    TrendContinuationReversal INT,
    MeanReversion INT,
    BollingerBands INT,
    MACD INT,
    SqueezeMomentum INT,
    CryptoLadder INT, 
    UNIQUE (Asset_ID),
    -- FOREIGN KEY (System_ID) REFERENCES SystemTable(System_ID) ON DELETE CASCADE,
    FOREIGN KEY (Asset_ID) REFERENCES Asset(Asset_ID) ON DELETE CASCADE
);

-- OrderTable
CREATE TABLE IF NOT EXISTS OrderTable (
    Order_ID INT AUTO_INCREMENT PRIMARY KEY,
    System_ID INT,
    -- Brokerage_ID INT,
    Asset_ID INT,
    Signal_ID INT,
    Order_Side VARCHAR(255),
    Price DECIMAL(10,2),
    Quantity INT,
    Timestamp TIMESTAMP,
    -- FOREIGN KEY (Brokerage_ID) REFERENCES Brokerage(Brokerage_ID) ON DELETE CASCADE,
    UNIQUE (Asset_ID),
    UNIQUE (Signal_ID),
    UNIQUE (System_ID),
    FOREIGN KEY (Asset_ID) REFERENCES Asset(Asset_ID) ON DELETE CASCADE,
    FOREIGN KEY (Signal_ID) REFERENCES SignalTable(Signal_ID) ON DELETE CASCADE,
    FOREIGN KEY (System_ID) REFERENCES SystemTable(System_ID) ON DELETE CASCADE
    -- UNIQUE (System_ID, Asset_ID)
);

-- Trade Table
CREATE TABLE IF NOT EXISTS Trade (
    Trade_ID INT AUTO_INCREMENT PRIMARY KEY,
    Order_ID INT,
    System_ID INT,
    Price DECIMAL(10,2),
    Quantity INT,
    Trade_Type VARCHAR(255),
    Timestamp TIMESTAMP, 
    UNIQUE (System_ID),
    FOREIGN KEY (Order_ID) REFERENCES OrderTable(Order_ID) ON DELETE CASCADE,
    FOREIGN KEY (System_ID) REFERENCES SystemTable(System_ID) ON DELETE CASCADE
);

-- PnL Table
CREATE TABLE IF NOT EXISTS PnL (
    PnL_ID INT AUTO_INCREMENT PRIMARY KEY,
    System_ID INT,
    Asset_ID INT,
    Realized_PnL DECIMAL(10,2),
    Unrealized_PnL DECIMAL(10,2),
    Timestamp TIMESTAMP, 
    UNIQUE (Asset_ID),
    UNIQUE (System_ID),
    FOREIGN KEY (Asset_ID) REFERENCES Asset(Asset_ID) ON DELETE CASCADE,
    FOREIGN KEY (System_ID) REFERENCES SystemTable(System_ID) ON DELETE CASCADE
);

-- Portfolio Table
CREATE TABLE IF NOT EXISTS Portfolio (
    Portfolio_ID INT AUTO_INCREMENT PRIMARY KEY,
    System_ID INT,
    Asset_ID INT,
    Quantity INT,
    Total_Value DECIMAL(10,2),
    Timestamp TIMESTAMP,
    UNIQUE (Asset_ID),
    UNIQUE (System_ID),
    FOREIGN KEY (Asset_ID) REFERENCES Asset(Asset_ID) ON DELETE CASCADE,
    FOREIGN KEY (System_ID) REFERENCES SystemTable(System_ID) ON DELETE CASCADE
);

-- HistoricalPrices Table
CREATE TABLE IF NOT EXISTS HistoricalPrices (
    Historical_Price_ID INT AUTO_INCREMENT PRIMARY KEY,
    Asset_ID INT,
    TimePeriod VARCHAR(255),
    DateIndex VARCHAR(255),
    Open DECIMAL(10,2),
    Close DECIMAL(10,2),
    High DECIMAL(10,2),
    Low DECIMAL(10,2),
    Volume INT,
    Timestamp TIMESTAMP,
    FOREIGN KEY (Asset_ID) REFERENCES Asset(Asset_ID) ON DELETE CASCADE
);
