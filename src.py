import pandas as pd
import numpy as np
from string import punctuation, digits
import re
import keras
import spacy
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings
warnings.filterwarnings('ignore')


def load_text():
    df = pd.read_csv('samples_no_title.csv',
                       skipinitialspace=True,
                       sep=',', 
                       quotechar='"', 
                       escapechar='\\',
                       error_bad_lines=False,
                       usecols = ['Grade','Text']).dropna()
    return df

def prepare_text(df, spacy_model=None):
    df = df.copy()
    df.Text = df.Text.str.replace('\n',' ')
    try:
        docs = df.Text.apply(spacy_model)
    except:
        spacy_model = spacy.load('en_core_web_lg')
        docs = df.Text.apply(spacy_model)
        pass

    lemmas = []
    grammar = []
    for doc in docs:
        lemma_text = ' '.join([sent.lemma_ for sent in doc.sents])
        lemmas.append(lemma_text)
        grammar_text = ' '.join([f'{word.text} {word.pos_} {word.dep_}' for word in doc])
        grammar.append(grammar_text)
    
    df['Lemmas'] = lemmas
    df['Grammar'] = grammar
    return df

def separate_sentences(df_in):
    if type(df_in) == str:
        df = df_in
        df = re.split('\;|\n|\.|\?|\!', df)
        df = pd.DataFrame(df, columns=['Text'])
        df = df[df.Text.str.len() > 4].dropna().reset_index(drop=True)
    else:
        df = df_in.copy()
        df.Text = df.Text.str.split(pat='\;|\n|\.|\?|\!', expand=False)
        df = df.explode('Text').dropna()
        df = df[df.Text.str.len() > 4].reset_index(drop=True)
    return df

def unique_characters(series):
    unique = set([])
    for text in series:
        characters = set(text)
        unique = unique.union(characters)
    return unique

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

def predict_grade(model, corpus, spacy_model=None):
    if type(corpus) == str:
        corpus = separate_sentences(corpus)
        corpus = prepare_text(corpus, spacy_model)
        yhat = model.predict(corpus).mean()
        return yhat
    else:
        predictions = []
        for text in corpus:
            predictions.append(predict_grade(model, text))
        return pd.Series(predictions)