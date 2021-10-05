import io
import tkinter
import PySimpleGUI as sg

from PIL import Image, ImageTk

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

# Initialize TensorFlow session.
tf.compat.v1.InteractiveSession()

# Import official CelebA-HQ networks.
with open('./karras2018iclr-celebahq-1024x1024.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

menu_def = [['File', ['Open', 'Save', 'Exit',]],
            ['Edit', ['Paste', ['Special', 'Normal',], 'Undo'],],
            ['Help', 'About...'],]

features = ['Attractive', 'Bald', 'Bangs', 'Big_Nose',
            'Eyeglasses', 'Gray_Hair', 'Heavy_Makeup', 'Male', 
            'Mustache', 'Smiling', 'Young', 'Wearing_Necktie']

NUM_ROWS = 4
NUM_COLS = 3

def read_axis(model='log_reg', orto=False, norm=True):
    filename = 'feature_axis.csv'
    if orto and norm:
        filename = 'feature_axis_orthonorm.csv'
    elif norm:
        filename = 'feature_axis_norm.csv'
     
    feature_axis = pd.read_csv(filename, index_col=0)
    # print(feature_axis.head())
    
    return feature_axis

def get_image_from_latent(latent, G, D, Gs):
    # Generate dummy labels (not used by the official networks).
    labels = np.zeros([latent.shape[0]] + Gs.input_shapes[1][1:])

    # Run the generator to produce a set of images.
    images = Gs.run(latent, labels)

    # Convert images to PIL-compatible format.
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1) # NCHW => NHWC
    
    return images[0]

def Euler_method(latent, h, tangent):
    tangent = tangent / np.linalg.norm(tangent)
    return latent + h * tangent

def linear_step(latent, feature_axis, feature='Male', h=1.0):
    W = feature_axis.loc[feature].values
    return Euler_method(latent, h, W)

def logistic_step(latent, feature_axis, feature='Male', h=1.0):
    W = feature_axis.loc[feature].values
    
    e_n_X_W = np.exp((-1) * np.dot(latent, W))[0]
    tangent = ((1.0 / ((1.0 + e_n_X_W)**2)) * e_n_X_W) * W
    
    return Euler_method(latent, h, tangent)

def create_options():
    options = []
    for i in range(NUM_ROWS):
        row_features = []
        for j in range(NUM_COLS):
            feature = features[i * NUM_COLS + j]
            widget = sg.Frame('', [
                [
                    sg.Text(feature, size=(14, 1), font=('Helvetica 12'), justification='center'), 
                    
                ], 
                [
                    sg.Button('-', key=feature+'-neg', size=(4, 1)), 
                    sg.Button('+', key=feature+'-pos', size=(4, 1)),
                    sg.Text('', size=(3, 1), font=('Helvetica 12'), justification='center', key=feature+'-lbl')
                ]
            ])
            row_features.append(widget)
        options.append(row_features)

    return options

def create_layout():
    options = create_options()

    layout = [[
        sg.Menu(menu_def),
        sg.Frame('', [[
            sg.Image(key='-IMAGE-', size=(80, 50))
        ]],
        ),
        sg.Frame('Options',
            [
                [
                    sg.Checkbox('Normalization', font=('Helvetica 12'), enable_events=True, key='norm', default=True),
                    sg.Checkbox('Orthogonalization', font=('Helvetica 12'), enable_events=True, key='orto'),
                    sg.Text('Lambda value:', font=('Helvetica 12')),
                    sg.InputText(default_text='1.0', key='lambda', enable_events=True, size=(8,2))
                ],
                [
                    sg.Button('Random photo', size=(16, 2), key='rnd'),
                    sg.Button('Step back', size=(13, 2), key='back'),
                    sg.Button('Reset', size=(11, 2), key='reset'),
                    sg.Button('Exit', size=(11, 2), key='exit'),
                ],
                [
                    sg.Frame(
                        'Edit photo',
                        options
                    ),
                ],
            ],
        ),
    ]]

    return layout

def img_to_bio(img):
    image = Image.fromarray(np.uint8(img))
    image.thumbnail((400, 400))
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    return bio.getvalue()

def create_gui():
    sg.theme('Dark Blue 3')  # please make your windows colorful
    layout = create_layout()
    window = sg.Window('Identikit generator', layout, finalize=True)

    feature_axis = read_axis()
    # print(feature_axis.head())

    # Generate latent vectors.
    latent = np.random.randn(1, *Gs.input_shapes[0][1:]) # 1 random latent vector
    # print('Latent generated:', latent)
    
    # Get the corresponding image
    img = get_image_from_latent(latent, G, D, Gs)
    # print('Original image completed')
    bio = img_to_bio(img)
    window["-IMAGE-"].update(data=bio)

    h = 1.0
    history = []
    while True:  # Event Loop
        event, values = window.read()
        
        if event == sg.WIN_CLOSED or event == 'exit':
            break
        
        elif event == 'rnd':
            # Generate latent vectors.
            latent = np.random.randn(1, *Gs.input_shapes[0][1:]) # 1 random latent vector

            history = []

            # Get the corresponding image
            img = get_image_from_latent(latent, G, D, Gs)
            bio = img_to_bio(img)
            window["-IMAGE-"].update(data=bio)

            for feature in features:
                window[feature+'-lbl'].update(value='')

        elif event == 'orto' or event == 'norm':
            orto = window['orto'].get()
            norm = window['norm'].get()

            feature_axis = read_axis(model='log_reg', orto=orto, norm=norm)

        elif event == 'lambda':
            try:
                h = float(values['lambda'])
            except:
                h = 1.0
        
        elif event == 'back':
            if len(history) > 0:
                latent, img, feature, direction = history.pop()

                bio = img_to_bio(img)
                window["-IMAGE-"].update(data=bio)

                value = window[feature+'-lbl'].get()
                if value=='':
                    value = 0
                else:
                    value = int(value)

                if direction == 'neg':
                    value = value+1
                else:
                    value = value-1

                if value > 0:
                    window[feature+'-lbl'].update(value='+'+str(abs(value)), text_color='#32CD32')
                elif value < 0:
                    window[feature+'-lbl'].update(value='-'+str(abs(value)), text_color='red')
                else:
                    window[feature+'-lbl'].update(value='')

        elif event == 'reset':
            if len(history) > 0:
                latent, img, _, _ = history[0]
                history = []

                bio = img_to_bio(img)
                window["-IMAGE-"].update(data=bio)

                for feature in features:
                    window[feature+'-lbl'].update(value='')


        elif '-' in event:
            event_parts = event.split('-')
            feature = event_parts[0]
            direction = event_parts[1]

            history.append((latent, img, feature, direction))

            if direction == 'neg':
                latent = logistic_step(latent, feature_axis, feature=feature, h=-h)
            else:
                latent = logistic_step(latent, feature_axis, feature=feature, h=+h)

            # Get the corresponding image
            img = get_image_from_latent(latent, G, D, Gs)
            # print('Original image completed')

            bio = img_to_bio(img)
            window["-IMAGE-"].update(data=bio)

            value = window[feature+'-lbl'].get()
            if value=='':
                value = 0
            else:
                value = int(value)

            if direction == 'neg':
                value = value-1
            else:
                value = value+1

            if value > 0:
                window[feature+'-lbl'].update(value='+'+str(abs(value)), text_color='#32CD32')
            elif value < 0:
                window[feature+'-lbl'].update(value='-'+str(abs(value)), text_color='red')
            else:
                window[feature+'-lbl'].update(value='')

    window.close()

if __name__=='__main__':
    create_gui()
