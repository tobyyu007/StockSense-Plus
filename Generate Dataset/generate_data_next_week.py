import pandas as pd

def get_weekly_news(start_date, end_date):
    results = {"positive": [], "negative": []}

    # Load the news data
    news_data = pd.read_csv('AAPL_news_sentiment.csv')
    

    # Convert 'datetime' to a uniform 'YYYY-MM-DD' format using a lambda function
    news_data['datetime'] = news_data['datetime'].apply(lambda x: x.split(' ')[0])

    # Convert start_date and end_date to datetime if they are not already, then to date
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    # Filter the news data for the given week
    # Need to ensure 'datetime' column is again converted to datetime.date for comparison
    weekly_news = news_data[(pd.to_datetime(news_data['datetime']).dt.date >= start_date) & 
                            (pd.to_datetime(news_data['datetime']).dt.date <= end_date)]
    
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

weekly_data = pd.read_csv('weekly_data.csv')
weekly_data['Next_Percentage_Change'] = weekly_data['Percentage_Change'].shift(-1)
instruction = "You are a seasoned stock market analyst. Your task is to predict the companies' stock price movement next week based on this week's positive headlines and negative headlines. Give me answer in  {increase/decrease/flat} in {X}%"
output = pd.DataFrame({'Instruction': [], 'Input': [], 'Output': []})

for index, row in weekly_data.iterrows():
    startDate = row['Week_Start']
    endDate = row['Week_End']
    # startPrice = round(row['Week_Open'], 2)
    # endPrice = round(row['Week_Close'], 2)
    # change = abs(round(row['Percentage_Change'], 2))
    change = abs(round(row['Next_Percentage_Change'], 2)) if pd.notnull(row['Next_Percentage_Change']) else None
    
    # Initialize input text for both positive and negative news headlines
    news = get_weekly_news(startDate, endDate)
    positive_news_text = "\n".join([f"* {title.replace('...', '')}" for title in news['positive']])
    negative_news_text = "\n".join([f"* {title.replace('...', '')}" for title in news['negative']])

    if positive_news_text == "" and negative_news_text == "":
        continue
    
    # Prepare the input text based on the direction of the change
    if row['Next_Percentage_Change'] == 0:
        direction = ""
        input_text = f"Company news during this period are listed below:\n\nPositive Headlines:\n{positive_news_text}\n\nNegative Headlines:\n{negative_news_text}"
        output_text = f"flat"
    else:
        if row['Next_Percentage_Change'] > 0:
            direction = "increased"
        else:
            direction = "decreased"
        
        input_text = f"Company news during this period are listed below:\n\nPositive Headlines:\n{positive_news_text}\n\nNegative Headlines:\n{negative_news_text}\n\n"
        output_text = f"{direction} in {change}%"
    
    output = output._append({'Instruction': instruction, 'Input': input_text, 'Output': output_text}, ignore_index=True)

output.to_csv('output_with_next_week.csv', index=False)
print(output)