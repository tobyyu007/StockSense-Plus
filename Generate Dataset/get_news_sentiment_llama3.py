import ollama
from datasets import load_dataset
import pandas as pd
import tqdm

def get_sentiment(symbol):
    modelfile='''
    FROM /Users/jackchen/Downloads/Stocksense-Plus-Full-GGUF-unsloth.Q4_K_M.gguf
    SYSTEM You are a seasoned stock market analyst. Your task is to do sentiment analysis on news titles. Don't say other things, just give me news titles in builtin points. Give me answer in this format {positive/neutral/negative}
    TEMPLATE "{{ if .System }}<|start_header_id|>system<|end_header_id|>

    {{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

    {{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

    {{ .Response }}<|eot_id|>"
    PARAMETER num_keep 24
    PARAMETER stop <|start_header_id|>
    PARAMETER stop <|end_header_id|>
    PARAMETER stop <|eot_id|>
    '''

    model_name = 'stocksense-plus-full'
    ollama.create(model=model_name, modelfile=modelfile)

    # Load json data from disk
    dataset = pd.read_csv('./dataset/news/' + symbol + '.csv')

    userPrompts = []
    for title in dataset['title'].tolist():
        if title[-1].isalpha() == True:
            title = title + '.'
        userPrompts.append([{'role': 'user', 'content': title + " Don't say other things, just give me answer in {negative/neutral/positive} for your response"}])

    # Prompt the model with the news title
    sentiments = []
    for idx, prompts in enumerate(tqdm.tqdm(userPrompts)):
        response = ollama.chat(model=model_name, messages=prompts)
        generatedAns = response['message']['content'].lower()
        sentiments.append(generatedAns)

    dataset['sentiment'] = sentiments

    # Save the sentiment to a csv file
    dataset.to_csv('./dataset/news sentiment llama3/' + symbol + '_news_sentiment.csv', index=False)



# NOT IN USED
def generate_output(symbol):
    # get_sentiment(symbol)

    modelfile='''
    FROM /Users/jackchen/Downloads/Stocksense-Plus-Full-GGUF-unsloth.Q4_K_M.gguf
    SYSTEM You are a seasoned stock market analyst. Your task is to do sentiment analysis on news titles. Don't say other things, just give me news titles in builtin points. Give me answer in this format {Positive: News Title}{Positive: News Title}{Positive: News Title}{Positive: News Title}{Positive: News Title}{Negative: News Title}{Negative: News Title}{Negative: News Title}{Negative: News Title}{Negative: News Title}
    TEMPLATE "{{ if .System }}<|start_header_id|>system<|end_header_id|>

    {{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

    {{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

    {{ .Response }}<|eot_id|>"
    PARAMETER num_keep 24
    PARAMETER stop <|start_header_id|>
    PARAMETER stop <|end_header_id|>
    PARAMETER stop <|eot_id|>
    '''

    model_name = 'stocksense-plus-full'
    ollama.create(model=model_name, modelfile=modelfile)

    # Load json data from disk
    dataset = pd.read_csv('./dataset/news sentiment llama3/' + symbol + '_news_sentiment.csv')
    # filter out neutral sentiment
    dataset = dataset[dataset['sentiment'] != 'neutral']

    # split dataset datetime into date and time using lambda function to split between space
    dataset['date'] = dataset['datetime'].apply(lambda x: x.split(' ')[0])
    weekly_data = pd.read_csv('./dataset/week change/' + symbol + '_weekly_data.csv')

    userPrompts = []
    for index, row in weekly_data.iterrows():
        startDate = row['Week_Start']
        endDate = row['Week_End']
        change = abs(round(row['Percentage_Change'], 2))

        # get the news for the week
        # Extract the relevant titles based on the date range
        news_titles = dataset[(dataset['date'] >= startDate) & (dataset['date'] <= endDate)]['title']
        content = ""
        for news_title in news_titles:
            content += '* ' + news_title + '\n'

        userPrompts.append([{'role': 'user', 'content': content + ". Give me 5 most positive and negative news. Don't say other things, just give me news titles in builtin points. Give me answer in this format {Positive: News Title}{Positive: News Title}{Positive: News Title}{Positive: News Title}{Positive: News Title}{Negative: News Title}{Negative: News Title}{Negative: News Title}{Negative: News Title}{Negative: News Title}"}])

        for idx, prompts in enumerate(tqdm.tqdm(userPrompts)):
            response = ollama.chat(model=model_name, messages=prompts)
            generatedAns = response['message']['content'].lower()
            print(type(generatedAns))
            print(generatedAns)
        break

# symbols = ["GOOG"]
# for symbol in symbols:
#     get_sentiment(symbol)
# generate_output(symbol)