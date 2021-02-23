import pandas as pd
import re
import keras

import warnings
warnings.filterwarnings('ignore')

def separate_sentences(df):
    if type(df) == str:
        df = re.split('\;|\n|\.|\?|\!', df)
        df = pd.DataFrame(df, columns=['Text'])
        df = df[df['Text'] != ''].dropna().reset_index(drop=True)
    else:
        df.Text = df.Text.str.split(pat='\;|\n|\.|\?|\!', expand=False)
        df = df.explode('Text').dropna()
        df = df[df.Text != ''].reset_index(drop=True)
    return df

def lower_case(string):
    return string.lower()

def load_model():
    return keras.models.load_model('model-Bi-LSTM-best')

def predict_grade(model, text):
    text = separate_sentences(text)
    text.Text = text.Text.apply(lower_case)
    return model.predict(text.Text).mean()