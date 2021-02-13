################### Imports ######################

# General imports
import numpy as np ; np.random.seed(1) # for reproducibility
import pandas as pd
from skimage import io
import json
import os
import plotly.express as px
import random
import shutil
import pathlib
import joblib
pd.options.mode.chained_assignment = None

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
from dash.dependencies import Input, Output, State  # ClientsideFunction
import colorlover
from dash_canvas import DashCanvas
from dash_canvas.utils import array_to_data_url, parse_jsonstring
from dash.exceptions import PreventUpdate

# Force to use CPU for this app
# tf.config.set_visible_devices([], 'GPU')

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
            html.Br(),
            html.H1(
                "Handwritten digit classifier with LeNet-5",
                style={
                    'text-align': 'center',
                    'font-family': 'monaco, sans-serif'
                }
            ),
            html.H5("About this App"),
            html.Div(
                children="This app allows you to recognize digits (i.e., numbers from 0 to 9) "
                         "manually drawn on your screen. I created this simple app as an example of how to combine "
                         "the complexity of Machine Learning algorithms (such as Convolutional Neural Networks) "
                         "with the beauty and simpleness of more comprehensive tools (i.e., Python-Dash)."
            ),
            html.Div([
                html.A("My LinkedIn profile",
                       href='https://www.linkedin.com/in/guillermo-lahuerta-pi%C3%B1eiro-b9a58913a/',
                       target="_blank")
            ]),
            html.Div([
                html.A("GitHub repo",
                       href='https://github.com/Guille1899/Digit_recognition',
                       target="_blank")
            ]),
            html.Br(),
            html.H5("LeNet-5"),
            html.Div(
                children="Convolutional Neural Networks is the standard architecture of a neural network designed for "
                         "solving tasks associated with images (e.g., image classification). Some of the well-known "
                         "deep learning architectures for CNN are LeNet-5 (7 layers), GoogLeNet (22 layers), AlexNet "
                         "(8 layers), VGG (16–19 layers), and ResNet (152 layers). For this app, I used LeNet-5, "
                         "which has been successfully used on the MNIST dataset to identify handwritten-digit "
                         "patterns."
            ),
            html.Br(),
            html.H5("Data"),
            html.Div(
                children="The dataset used to train, validate and test the model, correpsond to the MNIST dataset. "
                         "It is composed by a training set of 60,000 examples, and a test set of 10,000 examples. "
                         "The digits have been pre-processed to be size-normalized and centered in a fixed-size "
                         "image of 28x28 pixels."
            ),
            html.Div([
                html.A("MNIST database", href='http://yann.lecun.com/exdb/mnist/', target="_blank")
            ])
        ],
    )




################### Model ######################

# Load model
lenet_5_model = keras.models.load_model('./model/digit_recognizer_lenet_5.h5')

# Load history objects
acc = joblib.load('./model/acc_lenet_5.h5')
val_acc = joblib.load('./model/val_acc_lenet_5.h5')
loss = joblib.load('./model/loss_lenet_5.h5')
val_loss = joblib.load('./model/val_loss_lenet_5.h5')

# Get number of epochs
epochs = range(len(acc))




################### User Interface ######################

# Layout
app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            children=[
                # html.Img(
                #     src=app.get_asset_url("plotly_logo.png"),
                #     style={'height': '10%', 'width': '10%'}
                # )
            ],
        ),
        # Header
        html.Div(
            id="header",
            children=[
                description_card(),
                html.Hr(),
            ],
        ),
        # Main body
        html.Div([
            # Left column
            html.Div(
                id="left-column",
                children=[
                    html.Div(
                        children=[
                            html.H5("Draw digit"),
                            html.Div(
                                children="Please, do not touch the limits of the black canvas!"
                            ),
                            html.Br(),
                            DashCanvas(
                                id='digit_drawn',
                                filename="./assets/image_0.png",
                                width=420,
                                height=420,
                                scale=1,
                                lineWidth=50,
                                lineColor='white',
                                tool="pencil",
                                zoom=1,
                                goButtonTitle='Predict',
                                hide_buttons=["zoom", "pan", "line", "pencil", "rectangle", "select"]
                            )
                        ],
                    ),
                    html.Br(),
                ],
                className="three columns"
            ),
            # Center column
            html.Div(
                id="center-column",
                children=[
                    html.Div(
                        children=[
                            html.H5("Prediction"),
                            html.Div(
                                children="The digit drawn in the left canvas is pre-processed to be size-normalized "
                                         "and centered with 28x28 pixels (see image on the right). In this way, the image is as similar as "
                                         "possible as the training instances obtained from the MNIST database."
                            ),
                            html.Br(),
                            html.Div(
                                children="Please, note that the intention of this app was never to provide a super accurate model, "
                                         "but rather show a dummy example of how to integrate a CNN with a webapp."
                            ),
                            html.Br(),
                            html.Div(id='predict-text', style={'font-weight': 'bold', 'font-size': '60px'}),
                            html.Br(),
                        ],
                    ),
                    html.Br(),
                ],
                className="three columns"
            ),
            # Right column
            html.Div(
                id="right-column",
                children=[
                    html.Div(
                        children=[
                            html.Br(),
                            html.Div(id='predict-canvas'),
                        ],
                    ),
                    html.Br(),
                ],
                className="five columns"
            ),
        ]),
        html.Br(),
        html.Br(),
    ],
)


################### Callbacks ######################

@app.callback(Output('predict-canvas', 'children'),
              Output('predict-text', 'children'),
              Input('digit_drawn', 'json_data'))
def update_data(string):
    if string:
        mask = parse_jsonstring(string).astype(int)
        mask = (mask * 255).astype(np.uint8)
        new_image = Image.fromarray(mask)
        new_image.save("./data/img_to_predict.png")
        return generate_prediction()
    else:
        raise PreventUpdate




################### Run the App ######################

# Run the server
if __name__ == "__main__":
    #app.run_server(debug=True, port=8080)  # Comment this line when launching from the AWS server
    app.run_server(debug=False, host='0.0.0.0', port=8080) # Uncomment this line when launching from the AWS server
