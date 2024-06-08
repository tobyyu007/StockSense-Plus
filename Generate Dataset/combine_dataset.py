import pandas as pd

# # combine stock price data
# symbol = 'AAPL'
# stock_data = pd.read_csv('./dataset/stock price/' + symbol + '_stock_price_old.csv')
# new_stock_data = pd.read_csv('./dataset/stock price/' + symbol + '_stock_price_new.csv')

# print(stock_data.shape)
# print(new_stock_data.shape)

# # Combine the two datasets
# combined_stock_data = pd.concat([stock_data, new_stock_data], axis=0)
# print(combined_stock_data.shape)
# combined_stock_data.to_csv('./dataset/stock price/' + symbol + '_stock_price.csv', index=False)



# # combine news data
# symbol = 'AMZN'
# old_news_data = pd.read_csv('./dataset/news/' + symbol + '_old.csv')
# new_news_data = pd.read_csv('./dataset/news/' + symbol + '_new.csv')

# print(old_news_data.shape)
# print(new_news_data.shape)

# # Combine the two datasets
# combined_news_data = pd.concat([old_news_data, new_news_data], axis=0)
# print(combined_news_data.shape)
# combined_news_data.to_csv('./dataset/news/' + symbol + '.csv', index=False)



# # combine news sentiment data
# symbol = 'AMZN'
# news_data = pd.read_csv('./dataset/news sentiment/' + symbol + '_news_sentiment_old.csv')
# new_news_data = pd.read_csv('./dataset/news sentiment/' + symbol + '_news_sentiment_new.csv')

# print(news_data.shape)
# print(new_news_data.shape)

# # Combine the two datasets
# combined_news_data = pd.concat([news_data, new_news_data], axis=0)
# print(combined_news_data.shape)
# combined_news_data.to_csv('./dataset/news sentiment/' + symbol + '_news_sentiment.csv', index=False)



# # combine output - different companies with current week
# AAPL_current_week = pd.read_csv('./dataset/output/AAPL_output_with_current_week.csv')
# MSFT_current_week = pd.read_csv('./dataset/output/MSFT_output_with_current_week.csv')
# # GOOG_current_week = pd.read_csv('./dataset/output/GOOG_output_with_current_week.csv')
# AMZN_current_week = pd.read_csv('./dataset/output/AMZN_output_with_current_week.csv')

# print(AAPL_current_week.shape)
# print(MSFT_current_week.shape)
# # print(GOOG_current_week.shape)
# print(AMZN_current_week.shape)

# # Combine the two datasets
# # combined_data = pd.concat([AAPL_current_week, MSFT_current_week, GOOG_current_week, AMZN_current_week], axis=0)
# combined_data = pd.concat([AAPL_current_week, MSFT_current_week, AMZN_current_week], axis=0)
# print(combined_data.shape)
# combined_data.to_csv('./dataset/output/3stock_output_with_current_week.csv', index=False)



# combine output - different companies with current week_llama3
def combine_output_current_week_llama3(symbols):
    # Initialize an empty list to hold the dataframes
    dataframes = []

    for symbol in symbols:
        df = pd.read_csv(f'./dataset/output/{symbol}_output_with_current_week_llama3.csv')
        
        # Append the dataframe to the list
        dataframes.append(df)
        
        # Print the shape of the dataframe
        print(f'{symbol}_current_week.shape:', df.shape)

    # Concatenate all dataframes in the list
    combined_data = pd.concat(dataframes, axis=0)
    
    # Print the shape of the combined dataframe
    print('combined_data.shape:', combined_data.shape)
    
    # Save the combined dataframe to a new CSV file
    combined_data.to_csv(f'./dataset/output/{len(symbols)}stock_output_with_current_week_llama3.csv', index=False)
# Example Usage for combine_output_current_week_llama3
# symbols = ['AAPL', 'MSFT', 'AMZN']
# combine_output_current_week_llama3(symbols)



# combine output - different companies with current week
AAPL_current_week = pd.read_csv('./dataset/output/AAPL_output_with_current_week_llama3.csv')
MSFT_current_week = pd.read_csv('./dataset/output/MSFT_output_with_current_week_llama3.csv')
four_stock_current_week = pd.read_csv('./dataset/output/4stock_output_with_current_week_llama3.csv')

print(AAPL_current_week.shape)
print(MSFT_current_week.shape)
print(four_stock_current_week.shape)

# Combine the two datasets
# combined_data = pd.concat([AAPL_current_week, MSFT_current_week, GOOG_current_week, AMZN_current_week], axis=0)
combined_data = pd.concat([AAPL_current_week, MSFT_current_week, four_stock_current_week], axis=0)
print(combined_data.shape)
combined_data.to_csv('./dataset/output/6stock_output_with_current_week.csv', index=False)


# # combine output - current with next week
# current_week = pd.read_csv('./dataset/output/output_with_current_week.csv')
# next_week = pd.read_csv('./dataset/output/output_with_next_week.csv')

# print(current_week.shape)
# print(next_week.shape)

# # Combine the two datasets
# combined_data = pd.concat([current_week, next_week], axis=0)
# print(combined_data.shape)
# combined_data.to_csv('./dataset/output/output_AAPL.csv', index=False)



# # Combine the news data from different time periods
# symbol = 'META'
# news_2017_2018 = pd.read_csv('./dataset/' + symbol + ' (2017-01-01 to 2018-01-01).csv')
# news_2018_2019 = pd.read_csv('./dataset/' + symbol + ' (2018-01-01 to 2019-01-01).csv')
# news_2019_2020 = pd.read_csv('./dataset/' + symbol + ' (2019-01-01 to 2020-01-01).csv')
# news_2020_2021 = pd.read_csv('./dataset/' + symbol + ' (2020-01-01 to 2021-01-01).csv')
# news_2021_2022 = pd.read_csv('./dataset/' + symbol + ' (2021-01-01 to 2022-01-01).csv')
# news_2022_2023 = pd.read_csv('./dataset/' + symbol + ' (2022-01-01 to 2023-01-01).csv')
# news_2023_2024 = pd.read_csv('./dataset/' + symbol + ' (2023-01-01 to 2024-05-09).csv')
# # news_2023_2024 = pd.read_csv('./dataset/' + symbol + ' (2023-01-01 to 2024-01-01).csv')
# # news_2024_2024 = pd.read_csv('./dataset/' + symbol + ' (2024-01-01 to 2024-05-09).csv')

# print(news_2017_2018.shape)
# print(news_2018_2019.shape)
# print(news_2019_2020.shape)
# print(news_2020_2021.shape)
# print(news_2021_2022.shape)
# print(news_2022_2023.shape)
# print(news_2023_2024.shape)
# # print(news_2024_2024.shape)

# # Combine the datasets
# combined_news_data = pd.concat([news_2017_2018, news_2018_2019, news_2019_2020, news_2020_2021, news_2021_2022, news_2022_2023, news_2023_2024], axis=0)
# combined_news_data.to_csv('./dataset/' + symbol + '.csv', index=False)

# # drop duplicates
# combined_news_data = combined_news_data.drop_duplicates(subset=['title'])

# # drop not meaningful media sources
# meaningful_media = ['Forbes', 'TIME', 'The Wall Street Journal', 'Fox Business', 'NBC New York', 'NBC Bay Area', 'Yahoo News', 'NBC Boston'
# 'Seeking Alpha', 'Bloomberg', 'MarketWatch', 'USA TODAY', 'The New York Times', 'Yahoo News UK', 'Reuters', 'Morningstar', 'WSJ',
# 'Business Wire', 'CNBC', 'Business Insider', 'Minneapolis / St. Paul Business Journal', 'The Business Journals', 'Fortune', 'S&P Global', 'BuzzFeed News', 'Financial Times',
# 'The Motley Fool', 'Yahoo Finance', 'The Guardian', 'WIRED', 'The Economic Times', 'The Washington Post', 'Business Insider India', 'San Francisco Chronicle',
# 'NBC News', 'CBS News', 'Focus Taiwan', 'Barron\'s', 'International Business Times', 'Los Angeles Business Journal', 'Benzinga', 'BBC.com', 'Investopedia', 'Financial Post',
# 'Bitcoin Market Journal', 'Crunchbase News', 'Investing.com', 'Quartz', 'ABC News', 'Money Talks News', 'Yahoo Finance UK', 'Business in Vancouver', 'McKinsey',
# 'Small Business Trends', 'The Motley Fool Canada', 'The Financial Express', 'The Indian Wire', 'CBS New York', 'Washington Business Journal',
# 'Nasdaq', 'BNN Bloomberg', 'Business Today', 'Boston Business Journal', 'Fox News', 'CNN', 'Yahoo Finance Australia', 'Bloomberg Law', 'The Guardian US',
# 'New York Business Journal', 'NBC Los Angeles', 'Investing.com India', 'Finimize', 'Yahoo Money', 'TipRanks', 'Business.com', 'The Motley Fool Australia',
# 'Investing.com Australia']
# combined_news_data = combined_news_data[combined_news_data['media'].isin(meaningful_media)]

# print(combined_news_data.shape)
# combined_news_data.to_csv('./dataset/' + symbol + '.csv', index=False)