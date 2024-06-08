# from fetchNewsAPI import GoogleNews
from GoogleNews import GoogleNews
import time
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from random import randint
import csv
import pandas as pd
import urllib
import json
from multiprocessing import Pool, Manager
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import os


def getCompanyName(symbol):
    while True:
        try:
            response = urllib.request.urlopen(f'https://query2.finance.yahoo.com/v1/finance/search?q={symbol}')
            content = response.read()
            data = json.loads(content.decode('utf8'))['quotes'][0]['shortname']
            companyNames[str(symbol)] = str(data)
        except Exception:
            #tqdm.write("Error on %s retrying" % symbol)
            continue
        break


def datespan(startDate, endDate, delta=timedelta(days=1)):
        currentDate = startDate
        while currentDate < endDate:
            yield currentDate
            currentDate += delta


def getNews(stockSymbol):
    companyName = companyNames[stockSymbol]
    f = open('./dataset/' + str(stockSymbol) + ' (2022-01-01 to 2023-01-01).csv', 'w')
    writer = csv.writer(f)
    header = ['title', 'datetime', 'link', 'media']
    writer.writerow(header)

    for day in datespan(date(2022, 1, 1), date(2023, 1, 1), delta=relativedelta(days=1)):
        endDay = day + relativedelta(days=1)
        if int(day.day) % 2 == 0:
            https = True
        else:
            https = False
        # googlenews = GoogleNews(lang='en', region='US', start=str(day.strftime(
        #     "%m/%d/%Y")), end=str(endDay.strftime("%m/%d/%Y")), https=https)
        # googlenews.get_news(str(companyName))
        googlenews = GoogleNews(lang='en', region='US', start=str(day.strftime("%m/%d/%Y")), end=str(endDay.strftime("%m/%d/%Y")))
        googlenews.get_news(str(companyName))
        results = googlenews.results()
        print("number of results", len(results))
        for result in results:
            dateResult = []
            dateResult.append(str(result['title']))
            dateResult.append(str(day) + " 00:00:00")
            dateResult.append("https://" + str(result['link']))
            dateResult.append(str(result['media']))
            writer.writerow(dateResult)


if __name__ == '__main__':
    # read csv file
    # stockSymbolData = pd.read_csv('../dataset/MidCap Stock Symbols.csv')
    # stockSymbols = list(stockSymbolData["Symbol"])
    stockSymbols = list(["MSFT"])

    # Get all company names
    print("Fetch all company names")
    companyNames = {}
    
    for stockSymbol in tqdm(stockSymbols):
        getCompanyName(stockSymbol)

    # Get news from Google News
    print("\nStart getting news")
    for stockSymbol in stockSymbols:
        getNews(stockSymbol)