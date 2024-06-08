import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import re

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    try:
        tag = nltk.pos_tag([word])[0][1][0].upper()
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def preprocess(raw_text: str) -> str:
    try:
        stop_words = set(stopwords.words('english'))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

    # Step 1: remove non-letter characters and convert the string to lower case
    letters_only_text: str = re.sub("[^a-zA-Z]", " ", raw_text)
    letters_only_text: str = letters_only_text.lower()

    # Step 2: tokenization -- split into words -> convert string into list ( 'hello world' -> ['hello', 'world'])
    words: list[str] = letters_only_text.split()

    # Step 3: remove stopwords
    cleaned_words = []
    for word in words:
        if word not in stop_words:
            cleaned_words.append(word)

    # Step 4: lemmatise words
    lemmas = []
    lemmatizer = WordNetLemmatizer()
    for word in cleaned_words:
        lemma = lemmatizer.lemmatize(word, get_wordnet_pos(word))
        lemmas.append(lemma)

    # Step 5: converting list back to string and return
    return " ".join(lemmas)


def merge_lexicons(vader, lm_dict):
    new_lexicon = {}
    for word, scores in vader.items():
        new_lexicon[word] = scores
        if word.upper() in lm_dict:
            new_lexicon[word] *= 2
    for word, scores in lm_dict.items():
        word = word.lower() if isinstance(word, str) else word
        if word not in new_lexicon:
            if scores['Positive'] > 0:
                new_lexicon[word] = 4
            elif scores['Negative'] > 0:
                new_lexicon[word] = -4
            else:
                new_lexicon[word] = 0

    return new_lexicon


def get_sentiment_score(texts):
    scores = []

    vader = SentimentIntensityAnalyzer().lexicon
    lm_dict = pd.read_csv('./dataset/nlp/Loughran-McDonald_MasterDictionary_1993-2021.csv').set_index(
        'Word').to_dict('index')

    new_lexicon = merge_lexicons(vader, lm_dict)

    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon = new_lexicon

    for text in texts:
        processed_text = preprocess(text)
        score = analyzer.polarity_scores(processed_text)
        scores.append(score)

    return scores

def get_sentiment_as_dataframe(symbol):
    df = pd.read_csv('./dataset/news/' + symbol + '.csv')

    # get sentiment score on each title
    try:
        scores = get_sentiment_score(df['title'])

        for i in range(len(scores)):
            df.loc[i, 'Negative'] = scores[i]['neg']
            df.loc[i, 'Neutral'] = scores[i]['neu']
            df.loc[i, 'Positive'] = scores[i]['pos']
            df.loc[i, 'Compound'] = scores[i]['compound']
        
        df.to_csv('./dataset/news sentiment/' + symbol + '_news_sentiment.csv', index=False)
    except:
        scores = None
        # Write to .txt file indicating not finished company with correct scores
        with open('./dataset/Not Finished_news_sentiment.txt', 'a') as file:
            file.write(str(symbol))

# get_sentiment_as_dataframe('MSFT')