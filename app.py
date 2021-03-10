import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from writing_predictor import predict_grade
from keras.models import load_model
import en_core_web_sm

# set initial app settings
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

#Load the Model
model = load_model('model-Bi-LSTM-best')
spacy = en_core_web_sm.load()
process = 'Grammar'

# display layout and components
app.layout = html.Div([
    html.H2('Evaluate Children\'s Writing Samples'),
    html.H4('Text to evaluate'),
    html.H5('The more you write, the more accurate it will be.'),
    dcc.Textarea(
        id='sample-text',
        value='Please enter a sample to evaluate',
        style={'width': '%100', 'height': 300},
    ),
    html.Div(id='sample-text-output', style={'whiteSpace': 'pre-line'})



@app.callback(
    Output('sample-text-output', 'children'),
    Input('sample-text', 'value')
)
def update_output(text):
    if len(text) > 1:
        return predict_grade(model, text, process, spacy)
    else:
        return 0



if __name__ == '__main__':
    app.run_server(debug=True)
