1. **Brokerage Table**: Create a table for brokerages. It could contain columns like `Brokerage_ID`, `Brokerage_Name`, etc. This table will allow you to keep track of which brokerages you're trading on.

2. **System Table**: This table could keep track of the different trading systems you have. It might include columns like `System_ID`, `System_Name`, `Brokerage_ID (Foreign Key)`, and possibly others depending on what information you want to keep about each system.

3. **Symbol Table**: Add a `Brokerage_ID (Foreign Key)` column to your Symbol table to specify which symbols are available on which brokerage. Different brokerages may have access to different symbols.

4. **Order Table**: Add `System_ID (Foreign Key)` and `Brokerage_ID (Foreign Key)` columns to your Order Table. This will allow you to know which system and brokerage were used to place each order.

5. **Trade Table and PnL Table**: Similar to the Order Table, you could also add `System_ID (Foreign Key)` and `Brokerage_ID (Foreign Key)` columns to these tables.

6. **Strategy Weights Table**: You might want to include the `System_ID (Foreign Key)` here as well, in case different systems use different strategy weights.

7. **Portfolio Table**: The `Portfolio Table` should also have a `Brokerage_ID (Foreign Key)` column because different brokerages may have different portfolios. 

Remember, the way you design your tables and which fields you include will depend heavily on your specific needs. It's a good idea to plan this out carefully, considering what queries you will need to run and what information you will need to retrieve.



1. **Asset Table**: You might want to add an `Asset Table` to keep track of asset specific data. This table can include columns like `Symbol`, `Asset_Class`, `Tradable (True/False)`, and any other asset specific data you want to track.

2. **PnL Table**: To track the Profit and Loss (PnL) for each asset, consider adding a `PnL Table`. This table can include fields such as `PnL_ID`, `Trade_ID (Foreign Key)`, `Timestamp`, `Realized_PnL`, `Unrealized_PnL`, etc.

3. **Strategy Weights Table**: To keep track of the evolution of your strategy weights over time, you could add a `Strategy Weights Table`. This could include `Timestamp`, `Symbol`, and a field for each strategy weight.

4. **Portfolio Table**: To record the state of the portfolio at any given time, consider creating a `Portfolio Table` that records the current holdings and their quantities. This table could have columns like `Timestamp`, `Symbol`, `Quantity`, and `Total_Value`.

5. **Historical Prices Table**: To store historical prices for backtesting and strategy refinement, consider having a `Historical Prices Table`. This could include columns like `Timestamp`, `Symbol`, `Open`, `Close`, `High`, `Low`, `Volume`.
