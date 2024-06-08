import pandas as pd
import time
import stock_price_processing, newsCleaner, nlp, get_news_sentiment_llama3, generate_data_current_week, combine_dataset

def generate_data_pipeline(symbol, use_llama3=True):
    # Get historical stock price data
    stock_price_processing.get_stock_price(symbol)
    print("Stock price data for " + symbol + " is fetched!!")

    # Compute weekly stock price change
    stock_price_processing.get_weekly_stock_change(symbol)
    print("Weekly stock price change for " + symbol + " is computed!!")

    # Fetch news data (To be implemented)

    # Drop missing datetime and duplicate rows
    newsCleaner.news_cleaner(symbol)
    print("News data for " + symbol + " is cleaned!!")

    # Select only the meaningful title from the news
    meaningful_media = ['Forbes', 'TIME', 'The Wall Street Journal', 'Fox Business', 'NBC New York', 'NBC Bay Area', 'Yahoo News', 'NBC Boston'
    'Seeking Alpha', 'Bloomberg', 'MarketWatch', 'USA TODAY', 'The New York Times', 'Yahoo News UK', 'Reuters', 'Morningstar', 'WSJ',
    'Business Wire', 'CNBC', 'Business Insider', 'Minneapolis / St. Paul Business Journal', 'The Business Journals', 'Fortune', 'S&P Global', 'BuzzFeed News', 'Financial Times',
    'The Motley Fool', 'Yahoo Finance', 'The Guardian', 'WIRED', 'The Economic Times', 'The Washington Post', 'Business Insider India', 'San Francisco Chronicle',
    'NBC News', 'CBS News', 'Focus Taiwan', 'Barron\'s', 'International Business Times', 'Los Angeles Business Journal', 'Benzinga', 'BBC.com', 'Investopedia', 'Financial Post',
    'Bitcoin Market Journal', 'Crunchbase News', 'Investing.com', 'Quartz', 'ABC News', 'Money Talks News', 'Yahoo Finance UK', 'Business in Vancouver', 'McKinsey',
    'Small Business Trends', 'The Motley Fool Canada', 'The Financial Express', 'The Indian Wire', 'CBS New York', 'Washington Business Journal',
    'Nasdaq', 'BNN Bloomberg', 'Business Today', 'Boston Business Journal', 'Fox News', 'CNN', 'Yahoo Finance Australia', 'Bloomberg Law', 'The Guardian US',
    'New York Business Journal', 'NBC Los Angeles', 'Investing.com India', 'Finimize', 'Yahoo Money', 'TipRanks', 'Business.com', 'The Motley Fool Australia', 'Investing.com Australia']
    news_data = pd.read_csv('./dataset/news/' + symbol + '.csv')
    news_data = news_data[news_data['media'].isin(meaningful_media)]
    news_data.to_csv('./dataset/news/' + symbol + '.csv', index=False)
    print("Meaningful news data for " + symbol + " is selected!!")
    
    if use_llama3:
        # Generate sentiment for the news using Llama3
        start = time.time()
        get_news_sentiment_llama3.get_sentiment(symbol)
        end = time.time()
        print("Time taken to generate sentiment for " + symbol + " using Llama3: " + time.strftime("%H:%M:%S", time.gmtime(end - start)))
    else:
        # Generate sentiment for the news using McDonald NLP
        start = time.time()
        nlp.get_sentiment_as_dataframe(symbol)
        end = time.time()
        print("Time taken to generate sentiment for " + symbol + " using McDonald NLP: " + time.strftime("%H:%M:%S", time.gmtime(end - start)))
    
    # generate output for the current week
    generate_data_current_week.generate_current_week_output(symbol, use_llama3)
    print("Output for the current week for " + symbol + " is generated!!")

    print("Data pipeline for " + symbol + " is completed!!")


symbols = ['AMZN', 'GOOG', 'COST', 'META']
use_llama3 = True
for symbol in symbols:
    generate_data_pipeline(symbol, use_llama3)

# Combine the indvidual datasets that just ran
combine_dataset.combine_output_current_week_llama3(symbols)