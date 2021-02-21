import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from writing_predictor import predict_grade, load_model

# set initial app settings
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

#Load the Model
model = load_model()

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
    # html.H4('Country'),
    # html.Div(id='country-value'),
    # dcc.Dropdown(
    #     id='dropdown-country',
    #     options=[{'label': i, 'value': i} for i in unique_countries],
    #     value='None'
    # ),
    # html.H4('State'),
    # html.Div(id='state-value'),
    # dcc.Dropdown(id='dropdown-state',
    #              options=[{'label': 'None', 'value': 'None'}],
    #              value='None'
    #              ),
    # html.H4('Graph'),
    # html.Div(id='graph')
])


@app.callback(
    Output('sample-text-output', 'children'),
    Input('sample-text', 'value')
)
def update_output(text):
    if len(text) > 1:
        return predict_grade(model, text)
    else:
        return 0

# set state options according to chosen country
# @app.callback([dash.dependencies.Output('dropdown-state', 'options'),
#                dash.dependencies.Output('dropdown-state', 'value')],
#               [dash.dependencies.Input('dropdown-country', 'value')])
# def add_states(country_value, df=full_df):
#     country_df = full_df.loc[full_df['CountryName'] == country_value]
#     state_df = country_df.loc[country_df['Jurisdiction']
#                               == 'STATE_TOTAL'].dropna()
#     global country
#     country = country_value
#     if len(state_df) > 0:
#         return [{'label': 'None', 'value': 'None'}] + [{'label': i, 'value': i}
#                                                        for i in sorted(state_df['RegionName'].dropna().unique())], 'None'
#     else:
#         return [{'label': 'None', 'value': 'None'}], 'None'


# create graph
# @app.callback(dash.dependencies.Output('graph', 'children'),
#               [dash.dependencies.Input('dropdown-state', 'value'),
#                dash.dependencies.Input('dropdown-country', 'value'),
#                dash.dependencies.Input('checklist-stat', 'value')])
# def display_value(state, country, stats, df=full_df):
#     if country != 'None':
#         return dcc.Graph(id='stat-graph',
#                          figure=graph_stat(df, state=state,
#                                            country=country,
#                                            stats=stats))


if __name__ == '__main__':
    app.run_server(debug=True)
