import pandas as pd

symbol = 'AMZN'
def news_cleaner(symbol):
    df = pd.read_csv('./dataset/news/' + symbol + '.csv')
    print("Shape of news: ", df.shape)

    datetime_missing = df['datetime'].isnull().sum()
    print("Number of row missing Datetime column in news: ", datetime_missing)

    # drop rows with missing datetime
    df = df.dropna(subset=['datetime'])
    print("Shape of news after droping missing datetime: ", df.shape)

    # drop duplicate rows based on title
    df = df.drop_duplicates(subset=['title'])
    print("Shape of news after droping duplicates based on title: ", df.shape)

    # write df back to csv
    df.to_csv('./dataset/news/' + symbol + '.csv', index=False)