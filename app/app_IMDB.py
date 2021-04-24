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




################### Functions ######################

def imageprepare(path_to_image):

    """"
    Function to preprocess a raw image and transform it to MNIST style
    """

    # Load image and get dimensions
    img = Image.open(path_to_image).convert('L')
    width = float(img.size[0])
    height = float(img.size[1])

    # Create new image
    newImage = Image.new('L', (28, 28), 'black')
    nwidth = int(round((20.0 / height * width), 0))
    img = img.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    wleft = int(round(((28 - nwidth) / 2), 0))
    newImage.paste(img, (wleft, 4))

    # Process image
    img_array = np.array(newImage.getdata())
    img_array = tf.reshape(img_array, [28, 28, 1])

    return img_array


def generate_prediction():

    """"
    Function to preprocess a raw image and transform it to MNIST style
    """

    # Get names of pictures
    use_case_pic = "./data/img_to_predict.png"

    # Load image
    img = imageprepare(use_case_pic)

    # Pre-process image
    input_img = keras.preprocessing.image.img_to_array(img)
    input_img = input_img / 255.
    input_img = input_img.reshape((1,) + input_img.shape)

    # Predict
    classes = lenet_5_model.predict(input_img)
    certainty = str(np.max(classes * 100).round(1)) + '%'
    prediction = np.argmax(classes, axis=1)

    # Plot image
    fig = px.imshow(tf.reshape(img, [28, 28]), binary_string=True)
    text = 'This is a ' + str(prediction[0]) + ' [' + certainty + ']'

    # Plot grid
    return html.Div([
        dcc.Graph(figure=fig)
    ]), text


# Create a brief description of the tool
def description_card():
    """
    return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            # Row 0
            dbc.Row([
                html.Br(),
                html.H1(
                    "Sentiment analysis: IMDB reviews",
                    style={
                        'text-align': 'center',
                        'font-family': 'monaco, sans-serif'
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
                                html.A("LinkedIn",
                                       href='https://www.linkedin.com/in/guillermo-lahuerta-pi%C3%B1eiro-b9a58913a/',
                                       target="_blank")
                            ]),
                            html.Div([
                                html.A("GitHub repo",
                                       href='https://www.linkedin.com/in/guillermo-lahuerta-pi%C3%B1eiro-b9a58913a/n',
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
                                         "accuracy of 99.03% and a test accuracy of 84.27%."
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
                                children="The dataset used to train this model, correpsond to the 'IMDB reviews' dataset. "
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
                    src=app.get_asset_url("accuracy.png"),
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
                    src="https://projector.tensorflow.org/?config=https://gist.githubusercontent.com/Guille1899/6185a0ed9d82bf371a984cf7c2ec8547/raw/688afac9a363f872036640cf6e8ddf2fa036c576/config.json",
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
        html.Hr(),
        # Use case
        html.Div(
            id="use-case",
            children=[
                html.H5("Try it yourself!"),
                html.Br(),

            ],
            style={'width': '100%', 'justify': "center",
                   'vertical-align': 'middle', 'textAlign': 'center'}
        ),
        html.Br(),
        html.Br(),
        html.Br(),

        # # Second row
        # html.Div([
        #         html.Div(
        #             children=[
        #                 html.H5("Prediction"),
        #                 html.Div(
        #                     children="The digit drawn in the left canvas is pre-processed to be size-normalized "
        #                              "and centered with 28x28 pixels (see image on the right). In this way, the image is as similar as "
        #                              "possible as the training instances obtained from the MNIST database."
        #                 ),
        #                 html.Br(),
        #                 html.Div(
        #                     children="Please, note that the intention of this app was never to provide a super accurate model, "
        #                              "but rather show a dummy example of how to integrate a CNN with a webapp."
        #                 ),
        #                 html.Br(),
        #                 # html.Div(id='predict-text', style={'font-weight': 'bold', 'font-size': '60px'}),
        #                 html.Br(),
        #             ],
        #         ),
        #     html.Br(),
        # ]),
        # # Third row
        # html.Div([
        #         html.Div(
        #             children=[
        #                 html.Br(),
        #                 # html.Div(id='predict-canvas'),
        #             ],
        #         ),
        #         html.Br(),
        # ]),
        # html.Br(),
        # html.Br(),
    ],
)


################### Callbacks ######################

# @app.callback(Output('predict-canvas', 'children'),
#               Output('predict-text', 'children'),
#               Input('digit_drawn', 'json_data'))
# def update_data(string):
#     if string:
#         mask = parse_jsonstring(string).astype(int)
#         mask = (mask * 255).astype(np.uint8)
#         new_image = Image.fromarray(mask)
#         new_image.save("./data/img_to_predict.png")
#         return generate_prediction()
#     else:
#         raise PreventUpdate






################### Run the App ######################

# Run the server
if __name__ == "__main__":
    #app.run_server(debug=True, port=8080)  # Comment this line when launching from the AWS server
    app.run_server(debug=False, host='0.0.0.0', port=8080) # Uncomment this line when launching from the AWS server
