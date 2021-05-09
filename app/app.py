################### Imports ######################

# General imports
import numpy as np ; np.random.seed(1) # for reproducibility
import pandas as pd
# from skimage import io
import json
import os
import plotly.express as px
import random
import shutil
import pathlib
import joblib
pd.options.mode.chained_assignment = None
import numpy as np ; np.random.seed(1) # for reproducibility
import os
import joblib
import io
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import random
import shutil
from datetime import datetime
from sklearn.model_selection import train_test_split
import platform
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# TensorFlow
import tensorflow as tf
from tensorflow import keras

# PIL
import PIL
from PIL import Image, ImageDraw, ImageFilter

# Dash imports
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State  # ClientsideFunction
import colorlover
from dash_canvas import DashCanvas
from dash_canvas.utils import array_to_data_url, parse_jsonstring
from dash.exceptions import PreventUpdate

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


# Create a brief description of the tool
def description_card():
    """
    return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
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
            html.Br(),
            html.Br(),
            # First row
            dbc.Row([
                dbc.Col(
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
                        ],
                    ), style={'display': 'inline-block', 'width': '30%', 'justify': "center",
                              'vertical-align': 'top', 'margin-left': '0.5em', 'margin-right': '0.5em'}
                ),
                dbc.Col(
                    # Center column
                    html.Div(
                        id="center-column",
                        children=[
                            html.H5("Sentiment Analysis"),
                            html.Div(
                                children="Sentiment analysis is a Natural Language Processing technique used to determine the "
                                         "'sentiment' of a corpus of text (e.g., whether the opinion expressed is either positive or "
                                         "negative). The model presented in this app, provides the following accuracies a train "
                                         "accuracy of 99.20% and a test accuracy of 83.83%."
                            ),
                        ],
                    ), style={'display': 'inline-block', 'width': '30%', 'justify': "center",
                              'vertical-align': 'top', 'margin-left': '0.5em', 'margin-right': '0.5em'}
                ),
                dbc.Col(
                    # Right column
                    html.Div(
                        id="right-column",
                        children=[
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
                        ],
                    ), style={'display': 'inline-block', 'width': '30%', 'justify': "center",
                              'vertical-align': 'top', 'margin-left': '0.5em', 'margin-right': '0.5em'}
                )
                ])
        ],
        style={'width': '100%', 'justify': "center", 'vertical-align': 'middle'}
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
        # Description body
        dbc.Row([
            description_card(),
            html.Hr(),
        ]),
        # Accuracy body
        html.Div(
            id="accuracy",
            children=[
                html.H5("Model accuracy"),
                html.Br(),
                html.Img(
                    src=app.get_asset_url("acc.png"),
                    style={'height': '75%', 'width': '75%', 'justify': "center",
                           'vertical-align': 'middle', 'textAlign': 'center'}
                )
            ],
            style={'width': '100%', 'justify': "center",
                   'vertical-align': 'middle', 'textAlign': 'center'}
        ),
        html.Hr(),

        # Embeddings body
        html.Div(
            id="embeds",
            children=[
                html.H5("Embeddings"),
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
        html.Br(),
    ],
)



################### Run the App ######################

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True, port=8080)  # Comment this line when launching from the AWS server
    #app.run_server(debug=False, host='0.0.0.0', port=2020) # Uncomment this line when launching from the AWS server
