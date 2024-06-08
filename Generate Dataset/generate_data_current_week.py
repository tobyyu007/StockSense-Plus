import pandas as pd
import stock_price_processing

def get_weekly_news(start_date, end_date, symbol):
    results = {"positive": [], "negative": []}

    # Load the news data
    news_data = pd.read_csv('./dataset/news sentiment/' + symbol + '_news_sentiment.csv')
    

    # Convert 'datetime' to a uniform 'YYYY-MM-DD' format using a lambda function
    news_data['datetime'] = news_data['datetime'].apply(lambda x: x.split(' ')[0])

    # Convert start_date and end_date to datetime if they are not already, then to date
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    # Filter the news data for the given week
    # Need to ensure 'datetime' column is again converted to datetime.date for comparison
    weekly_news = news_data[(pd.to_datetime(news_data['datetime'], errors='coerce').dt.date >= start_date) & 
                            (pd.to_datetime(news_data['datetime'], errors='coerce').dt.date <= end_date)]
    
    # Get the positive news titles for the week (Compound score >= 0.05)
    positive_news = weekly_news[weekly_news['Compound'] >= 0.05]
    # Sort by 'Compound' score in descending order and pick the top 5
    positive_news = positive_news.sort_values(by='Compound', ascending=False).head(5)
    results['positive'] = positive_news['title'].tolist()

    # Get the negative news titles for the week (Compound score <= -0.05)
    negative_news = weekly_news[weekly_news['Compound'] <= -0.05]
    # Sort by 'Compound' score in ascending order and pick the top 5
    negative_news = negative_news.sort_values(by='Compound').head(5)
    results['negative'] = negative_news['title'].tolist()
    
    return results


def get_weekly_news_llama3(start_date, end_date, symbol):
    results = {"positive": [], "negative": []}

    # Load the news data
    news_data = pd.read_csv('./dataset/news sentiment llama3/' + symbol + '_news_sentiment.csv')
    

    # Convert 'datetime' to a uniform 'YYYY-MM-DD' format using a lambda function
    news_data['datetime'] = news_data['datetime'].apply(lambda x: x.split(' ')[0])

    # Convert start_date and end_date to datetime if they are not already, then to date
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    # Filter the news data for the given week
    # Need to ensure 'datetime' column is again converted to datetime.date for comparison
    weekly_news = news_data[(pd.to_datetime(news_data['datetime'], errors='coerce').dt.date >= start_date) & 
                            (pd.to_datetime(news_data['datetime'], errors='coerce').dt.date <= end_date)]
    
    # Get the positive news titles for the week
    positive_news = weekly_news[weekly_news['sentiment'] == ' positive']
    if len(positive_news) > 5:
        positive_news = positive_news.sample(5)
    results['positive'] = positive_news['title'].tolist()
    negative_news = weekly_news[weekly_news['sentiment'] == ' negative']
    if len(negative_news) > 5:
        negative_news = negative_news.sample(5)
    results['negative'] = negative_news['title'].tolist()
    
    return results


def generate_current_week_output(symbol, use_llama3=True):
    stock_price_processing.get_weekly_stock_change(symbol)
    weekly_data = pd.read_csv('./dataset/week change/' + symbol + '_weekly_data.csv')
    instruction = "You are a seasoned stock market analyst. Your task is to predict the companies' stock price movement for this week based on this week's positive headlines and negative headlines. Give me answer in the format of {increased/decreased/flat} in {X}%"
    output = pd.DataFrame({'Instruction': [], 'Input': [], 'Output': []})

    for index, row in weekly_data.iterrows():
        startDate = row['Week_Start']
        endDate = row['Week_End']
        change = abs(round(row['Percentage_Change'], 2))
        
        # Initialize input text for both positive and negative news headlines
        if use_llama3:
            news = get_weekly_news_llama3(startDate, endDate, symbol)
        else:
            news = get_weekly_news(startDate, endDate, symbol)
        
        positive_news_text = "\n".join([f"* {title.replace('...', '')}" for title in news['positive']])
        negative_news_text = "\n".join([f"* {title.replace('...', '')}" for title in news['negative']])

        if positive_news_text == "" and negative_news_text == "":
            continue
        
        # Prepare the input text based on the direction of the change
        if row['Percentage_Change'] == 0:
            direction = ""
            input_text = f"Company news during this period are listed below:\n\nPositive Headlines:\n{positive_news_text}\n\nNegative Headlines:\n{negative_news_text}"
            output_text = f"flat"
        else:
            if row['Percentage_Change'] > 0:
                direction = "increased"
            else:
                direction = "decreased"
            
            input_text = f"Company news during this period are listed below:\n\nPositive Headlines:\n{positive_news_text}\n\nNegative Headlines:\n{negative_news_text}\n\nPredict {symbol}'s stock price movement for this week based on this week's positive headlines and negative headlines. Give me answer in the format of {{increased/decreased/flat}} in {{X}}%"
            output_text = f"{direction} in {change}%"
        
        output = output._append({'Instruction': instruction, 'Input': input_text, 'Output': output_text}, ignore_index=True)
    output.to_csv('./dataset/output/' + symbol + '_output_with_current_week_llama3.csv', index=False)
    print(output)
    print("Finished generating output for " + symbol)
    

# symbols = ['AAPL']
# use_llama3 = True
# for symbol in symbols:
#     generate_current_week_output(symbols, use_llama3)