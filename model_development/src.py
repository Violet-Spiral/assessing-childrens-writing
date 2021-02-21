import pandas as pd
import numpy as np
from string import punctuation, digits
import re
import keras
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer 
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings
warnings.filterwarnings('ignore')

def separate_sentences(df):
    if type(df) == str:
        df = re.split('\;|\n|\.|\?|\!', df)
        df = pd.DataFrame(df, columns=['Text'])
        df = df[df.Text.str.len() > 10].dropna().reset_index(drop=True)
    else:
        df.Text = df.Text.str.split(pat='\;|\n|\.|\?|\!', expand=False)
        df = df.explode('Text').dropna()
        df = df[df.Text.str.len() > 1].reset_index(drop=True)
    return df

def load_text(sentences=False, grammarize=False):
    df = pd.read_csv('../data/samples_no_title.csv',
                       skipinitialspace=True,
                       sep=',', 
                       quotechar='"', 
                       escapechar='\\',
                       error_bad_lines=False,
                       usecols = ['Grade','Text']).dropna()
    if grammarize:
        df.Text = grammar_text(df.Text)
    if sentences:
        df = separate_sentences(df)
        
    return df

def get_word_index(df):
    vocabulary = set([])
    for text in df.Text:
        text = text.replace('\n',' ')
        vocabulary = vocabulary.union([word.strip() for word in text.split()])
    word_index = dict(zip(vocabulary, range(len(vocabulary))))
    return word_index

def unique_characters(series):
    unique = set([])
    for text in series:
        characters = set(text)
        unique = unique.union(characters)
    return unique

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
    new_punct = punctuation.replace('*', "’-")
    for punct in new_punct:
        string = string.replace(punct,f' {punct} ').replace('  ',' ')
    return string

def tag_uppercase(series):
    repl = lambda w: f'+ {w.group(0).lower()}'
    return series.str.replace('[A-Z]\w+',repl,regex=True)

def remove_weird_chars(string):
    weird_chars = [ '¦', '©', '±', '³', '½', 'Â', 'Ã', 'â', '“', '”', '€','\n']
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

def lemmatize_grammarize_text(df):
    df['lemmatized'] = lemma_text(df.Text).str.join(' ')
    df['grammarized'] = grammar_text(df.Text)
    return df

def assess_model(model, X_train, y_train, scores=None, ngram_range=(1,3)):
    if not scores:
        scores = pd.DataFrame(columns = ['model','ngram','encoding'])
    for ngram in range(ngram_range[0],ngram_range[1]+1):
        lr_count_pipe = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,ngram))),
                                  ('logreg', model)])
        lr_tfidf_pipe = Pipeline([('vectorizer', TfidfVectorizer(ngram_range=(1,ngram))),
                              ('logreg', model)])
        tfidf_scores = cross_val_score(lr_tfidf_pipe, X_train, y_train, cv=3, 
                                         scoring='neg_mean_absolute_error')
        
        count_scores = cross_val_score(lr_count_pipe, X_train, y_train, cv=3, 
                                         scoring='neg_mean_absolute_error')
        scores = scores.append({'model':type(model).__name__,
                        'encoding':'Count Vectors',
                        'ngram':ngram,
                       'score':-np.mean(count_scores)},
                      ignore_index=True)

        scores = scores.append({'model':type(model).__name__,
                        'encoding':'TF-IDF Vectors',
                        'ngram':ngram,
                      'score':-np.mean(tfidf_scores)},
                      ignore_index=True)
        print('finished ngram', ngram)
        
    return scores

def load_model():
    return keras.models.load_model('best-MLP')

def predict_grade(model, corpus):
    if type(corpus) == str:
        corpus = separate_sentences(corpus)
        yhat = model.predict(corpus).mean()
        return yhat
    else:
        predictions = []
        for text in corpus:
            predictions.append(predict_grade(model, text))
        return pd.Series(predictions)