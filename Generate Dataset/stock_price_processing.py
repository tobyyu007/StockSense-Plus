import yfinance as yf
import datetime
import pandas as pd

start = datetime.datetime(2017, 1, 1)
end = datetime.datetime(2024, 5, 9)

def get_stock_price(symbol):
    stock = yf.download(symbol, start=start, end=end, progress=False)
    stock.to_csv("./dataset/stock price/" + symbol + "_stock_price.csv", index=True)


def get_weekly_stock_change(symbol):
    df = pd.read_csv('./dataset/stock price/' + symbol + '_stock_price.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    weekly_data = pd.DataFrame()

    # Find the start and end of the dataset to handle the range
    start_date = df.index.min()
    end_date = df.index.max()

    # Initialize the current week's start and end dates
    current_start = start_date - pd.Timedelta(days=start_date.weekday())
    current_end = current_start + pd.Timedelta(days=4)

    # Process weeks until the end of the dataset
    while current_start <= end_date:
        # Filter data to get the week's data from Monday to Friday
        weekly_df = df[(df.index >= current_start) & (df.index <= current_end)]
        
        if not weekly_df.empty:
            # Get the first and last 'Adj Close' of the week
            week_open = weekly_df['Adj Close'].iloc[0]
            week_close = weekly_df['Adj Close'].iloc[-1]
            
            # Calculate change and percentage change
            weekly_change = week_close - week_open
            percentage_change = (weekly_change / week_open) * 100
            
            # Append the results to the weekly_data DataFrame
            weekly_data = weekly_data._append({
                'Week_Start': current_start,
                'Week_End': current_end,
                'Week_Open': week_open,
                'Week_Close': week_close,
                'Weekly_Change': weekly_change,
                'Percentage_Change': percentage_change
            }, ignore_index=True)
        
        # Move to the next week
        current_start += pd.Timedelta(days=7)
        current_end += pd.Timedelta(days=7)

    weekly_data.to_csv('./dataset/week change/' + symbol + '_weekly_data.csv', index=False)

# symbol = 'AMZN'
# get_weekly_stock_change(symbol)