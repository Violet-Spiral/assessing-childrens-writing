import pandas as pd
import numpy as np
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from string import punctuation

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

import pickle

import warnings
warnings.filterwarnings('ignore')

def unique_characters(series):
    unique = set([])
    for text in series:
        characters = set(text)
        unique = unique.union(characters)
    return unique

#nltk_tag_to_wordnet_tag() courtesy of [Gaurav Gupta](https://medium.com/@gaurav5430/using-nltk-for-lemmatizing-sentences-c1bfff963258)
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None    
    
def retag(tagged_list):
    return [(w[0], nltk_tag_to_wordnet_tag(w[1])) for w in tagged_list]

def lemma(tagged_tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for w in tagged_tokens:
        if w[1] is None:
            lemmas.append(lemmatizer.lemmatize(w[0]))
        else:
            lemmas.append(lemmatizer.lemmatize(w[0],w[1]))                   
    return lemmas

def lower_case(string):
    return string.lower()

def letters(string):
    letters = re.findall('[a-z]+',string)
    return letters

def clean_tokenize(series):
    return series.apply(lower_case).apply(letters)

def lemmatize(series):
    return series.apply(pos_tag).apply(retag).apply(lemma)

def replace_urls(series):
    return series.str.replace('www.','',regex=False).replace('\w*\.\w{2,}', value = "*url", regex=True)

def isolate_punctuation(string):
    for punct in punctuation.replace('*',''):
        string = string.replace(punct,f' {punct} ').replace('  ',' ')
    return string

def tag_uppercase(series):
    repl = lambda w: f'+ {w.group(0).lower()}'
    return series.str.replace('[A-Z]\w+',repl,regex=True)

def remove_weird_chars(string):
    weird_chars = [ '¦', '©', '±', '³', '½', 'Â', 'Ã', 'â', '“', '”', '€']
    return ''.join([c for c in string if c not in weird_chars])

def grammar_text(series):
    series = replace_urls(series)
    series = series.apply(remove_weird_chars).apply(isolate_punctuation)
    series = tag_uppercase(series)
    return series

def lemma_text(series):
    series = clean_tokenize(series)
    series = lemmatize(series)
    return series

def process_text(df,strategy=None):
    if strategy == 'lemmatize':
        df = lemma_text(df.Text).str.join(' ')
    if strategy == 'grammarize':
        df = grammar_text(df.Text)
    else: 
        df['lemmatized'] = lemma_text(df.Text).str.join(' ')
        df['grammarized'] = grammar_text(df.Text)
    return df

def predict_grade(text, strategy='grammarize'):
    df = pd.DataFrame({'Text':[text,text]})
    processed_text = process_text(df, strategy)
    model = pickle.load(open('best_model.pkl','rb'))
    return model.predict([processed_text[0]])[0]