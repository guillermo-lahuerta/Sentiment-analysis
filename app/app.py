################### Imports ######################

# General imports
import numpy as np ; np.random.seed(1) # for reproducibility
import pandas as pd
import pathlib
pd.options.mode.chained_assignment = None
import numpy as np ; np.random.seed(1) # for reproducibility
import os
import joblib

# TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Dash imports
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State  # ClientsideFunction

# Indicate the version of Tensorflow and whether it uses the CPU or the GPU
print("TensorFlow version:", tf.__version__)
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("The GPU will be used for calculations.")
else:
    print("The CPU will be used for calculations.")




################### Dash set up ######################

# Set up
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
server = app.server
app.config.suppress_callback_exceptions = True

# Define paths
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()


################## Functions #####################

def predict_sentiment(text):

    # Set values
    max_length = 240
    trunc_type = 'post'

    # Tokenize
    tokenizer = joblib.load(os.path.join(path_model, 'tokenizer'))

    # # Use absolute path when running in the server
    # tokenizer = joblib.load('/home/ubuntu/Sentiment-analysis/app/tokenizer')

    # Sequence
    sequences = tokenizer.texts_to_sequences([text])

    # Add padding
    padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

    # Predict
    predictions = model.predict(padded)

    # Get response
    if predictions[0] < 0.5:
        response = "BAD"
    else:
        response = "GOOD"

    return response


# Create a brief description of the tool
def description_card():
    """
    return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="title-card",
        children=[
            dbc.Row([
                html.Br(),
                html.H1(
                    "Sentiment analysis: IMDB reviews",
                    style={
                        'text-align': 'center',
                        'font-family': 'verdana, sans-serif',
                        'color': '#f3ce13'
                    }
                ),
            ]),
        ]
    )


################### Paths ######################

# Define paths
path_data = '../data'
path_model = '../model'
path_output = '../output'


#################### Loads #####################

# Load model and history
model = keras.models.load_model(os.path.join(path_model, 'imdb_model.h5'))
model.load_weights(os.path.join(path_model, 'imdb_weights.h5'))
history_dict = joblib.load(os.path.join(path_model, 'imdb_history'))

# # Use absolute path when running in the server
# model = keras.models.load_model('/home/ubuntu/Sentiment-analysis/app/imdb_model.h5')
# model.load_weights('/home/ubuntu/Sentiment-analysis/app/imdb_weights.h5')
# history_dict = joblib.load('/home/ubuntu/Sentiment-analysis/app/imdb_history')




################### User Interface ######################

# Layout
app.layout = html.Div(
    id="app-container",
    children=[

        # Banner
        html.Div(
            id="banner",
            children=[
                html.Img(
                    src=app.get_asset_url("imdb_logo.jpeg"),
                    style={'height': '5%', 'width': '5%'}
                )
            ],
        ),

        # Title body
        dbc.Row([
            description_card(),
            html.Hr(),
        ]),

        # Description body
        dbc.Row([
            dbc.Col([
                html.Div(
                    # Left column
                    html.Div(
                        id="left-column",
                        children=[
                            html.H5("About this App"),
                            html.Div(
                                children="This app allows you to classify movie reviews extracted from IMBD. "
                                         "By means of embeddings, it also allows you to visualize how the different words cluster with each other."
                            ),
                            html.Div([
                                html.A("GitHub repo",
                                       href='https://github.com/guillermo-lahuerta/Sentiment_analysis',
                                       target="_blank")
                            ]),
                            html.Br(),
                            html.H5("Data"),
                            html.Div(
                                children="The dataset used to train this model, correpsonds to the 'IMDB reviews' dataset. "
                                         "It is composed by a training set of 25,000 examples, and a test set of 25,000 examples. "
                            ),
                            html.Div([
                                html.A("IMDB dataset",
                                       href='https://www.tensorflow.org/datasets/catalog/imdb_reviews/',
                                       target="_blank")
                            ]),
                            html.Br(),
                            html.Div(
                                id="center-column",
                                children=[
                                    html.H5("Sentiment Analysis"),
                                    html.Div(
                                        children="Sentiment analysis is a Natural Language Processing technique used to determine the "
                                                 "'sentiment' of a corpus of text (e.g., whether the opinion expressed is either positive or "
                                                 "negative). The model presented in this app, provides the following accuracies a train "
                                                 "accuracy of 95.23% and a test accuracy of 83.88%."
                                    ),
                                ],
                            )
                        ],
                    ), style={'display': 'inline-block', 'width': '50%', 'justify': "center",
                              'vertical-align': 'top', 'margin-left': '0.5em', 'margin-right': '0.5em',
                              'textAlign': 'center'}
                )
            ]),

            # Accuracy body
            html.Br(),
            html.Hr(),
            html.Div(
                id="accuracy",
                children=[
                    html.H5("Model evaluation"),
                    html.Br(),
                    html.Img(
                        src=app.get_asset_url("acc.png"),
                        style={'height': '75%', 'width': '75%', 'justify': "center",
                               'vertical-align': 'middle', 'textAlign': 'center'}
                    )], style={'width': '100%', 'justify': "center", 'vertical-align': 'middle', 'textAlign': 'center'}
            ),
            html.Hr(),
        ], style={'width': '100%', 'justify': "center", 'vertical-align': 'middle', 'textAlign': 'center'}
        ),

        # Embeddings body
        html.Div(
            id="embeds",
            children=[
                html.H5("Embeddings"),
                html.Br(),
                html.Br(),
                html.B('Please, click on "Sphereize data" to normalise the data and see the proper clusters (the '
                       'checkbox option is on the left hand side).'),
                html.Br(),
                html.Br(),
                html.Iframe(
                    src="https://projector.tensorflow.org/?config=https://gist.githubusercontent.com/guillermo-lahuerta/6185a0ed9d82bf371a984cf7c2ec8547/raw/688afac9a363f872036640cf6e8ddf2fa036c576/config.json",
                    width="1500",
                    height="600"
                )],
            style={'display': 'inline-block', 'justify': "center", 'width': '100%', 'textAlign': 'center'}
        ),
        html.Hr(),

        # Word cloud body
        html.Div(
            id="wordcloud",
            children=[
                html.H5("Word cloud"),
                html.Br(),
                html.Img(
                    src=app.get_asset_url("wordcloud.png"),
                    style={'height': '35%', 'width': '35%', 'justify': "center",
                           'vertical-align': 'middle', 'textAlign': 'center'}
                )
            ],
            style={'width': '100%', 'justify': "center",
                   'vertical-align': 'middle', 'textAlign': 'center'}
        ),
        html.Br(),

        # Write canvas
        html.Hr(),
        html.Div(
            id="canvas",
            children=[
                html.H5("Try it yourself!"),
                html.Br(),
                dcc.Textarea(
                    id='textarea-state',
                    value='Game of Thrones is awesome',
                    style={'width': '60%', 'height': 50},
                ),
                html.Br(),
                html.Br(),
                html.Button('Predict sentiment', id='textarea-state-button', n_clicks=0,
                            style={'background-color': '#4CAF50', 'color': 'white'}),
                html.Div(id='textarea-state-output', style={'whiteSpace': 'pre-line'})
            ],
            style={'width': '100%', 'justify': "center",
                   'vertical-align': 'middle', 'textAlign': 'center'}
        ),
        html.Br(),
        html.Br(),
    ],
)


################### Callbacks ######################

@app.callback(
    Output('textarea-state-output', 'children'),
    Input('textarea-state-button', 'n_clicks'),
    State('textarea-state', 'value')
)
def update_output(n_clicks, value):
    if n_clicks > 0:
        resp = predict_sentiment(value)
        return 'The expected sentiment is: \n{}'.format(resp)



################### Run the App ######################

# Run the server
if __name__ == "__main__":
    # app.run_server(debug=True, port=80)  # Comment this line when launching from the AWS server
    app.run_server(debug=False, host='0.0.0.0', port=8082) # Uncomment this line when launching from the AWS server
