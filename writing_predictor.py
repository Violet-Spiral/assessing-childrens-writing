import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

def load_text():
    import pandas as pd
    df = pd.read_csv('samples_no_title.csv',
                       skipinitialspace=True,
                       sep=',', 
                       quotechar='"', 
                       escapechar='\\',
                       error_bad_lines=False,
                       usecols = ['Grade','Text']).dropna()
    return df

def separate_sentences(df_in):
    if type(df_in) == str:
        df = df_in
        df = re.split('\;|\n|\.|\?|\!', df)
        df = pd.DataFrame(df, columns=['Text'])
        df = df[df.Text.str.len() > 0].dropna().reset_index(drop=True)
    else:
        df = df_in.copy()
        df.Text = df.Text.str.split(pat='\;|\n|\.|\?|\!', expand=False)
        df = df.explode('Text').dropna()
        df = df[df.Text.str.len() > 4].reset_index(drop=True)
    return df

def prepare_text(df, spacy_model, process='Lemmas'):
    if type(df) == str:
        df = pd.DataFrame([df], columns = ['Text'])
    df.Text = df.Text.str.replace('\n',' ')
    docs = df.Text.apply(spacy_model)
    newtext = []
    if process == 'Grammar':
        for doc in docs:
            grammar_text = ' '.join([f'{word.text} {word.pos_} {word.dep_}' for word in doc])
            newtext.append(grammar_text)
    elif process == 'Lemmas':
        for doc in docs:
            lemma_text = ' '.join([sent.lemma_ for sent in doc.sents])
            newtext.append(lemma_text)
    else:
        print('Choose "Lemmas" or "Grammar"')
        return None
    
    df[process] = newtext
    return df

def predict_grade(model, corpus, process, spacy_model):
    if type(corpus) == str:
        corpus = separate_sentences(corpus)
        corpus = prepare_text(corpus, spacy_model, process)
        yhat = model.predict(corpus[process]).mean()
        return yhat
    else:
        predictions = []
        for text in corpus:
            predictions.append(predict_grade(model, text, process, spacy_model))
        return pd.Series(predictions)