import io
import tkinter
import PySimpleGUI as sg

from PIL import Image, ImageTk

# root = tkinter.Tk()

sg.theme('Dark Blue 3')  # please make your windows colorful

menu_def = [['File', ['Open', 'Save', 'Exit',]],
            ['Edit', ['Paste', ['Special', 'Normal',], 'Undo'],],
            ['Help', 'About...'],]

features = ['Attractive', 'Bald', 'Bangs', 'Big_Nose',
            'Eyeglasses', 'Gray_Hair', 'Heavy_Makeup', 'Male', 
            'Mustache', 'Smiling', 'Young', 'Wearing_Necktie']

NUM_ROWS = 4
NUM_COLS = 3

options = []

for i in range(NUM_ROWS):
    row_features = []
    for j in range(NUM_COLS):
        feature = features[i * NUM_COLS + j]
        widget = sg.Frame('', [
            [sg.Text(feature, size=(14, 1), font=('Helvetica 12'), justification='center')], 
            [sg.Button('-', key=feature+'-neg', size=(7, 1)), sg.Button('+', key=feature+'-pos', size=(7, 1))]])
        row_features.append(widget)
    options.append(row_features)

layout = [[
    sg.Menu(menu_def),
    sg.Frame('', [[
        sg.Image(key='-IMAGE-', size=(80, 50))
    ]],
    ),
    sg.Frame('Options',
        [
            [
                sg.Checkbox('Normalization', font=('Helvetica 12')),
                sg.Checkbox('Orthogonalization', font=('Helvetica 12')),
                sg.Text('Lambda value:', font=('Helvetica 12')),
                sg.InputText(size=(11,2))
            ],
            [
                sg.Button('Random photo', size=(16, 2)),
                sg.Button('Step back', size=(14, 2)),
                sg.Button('Reset', size=(12, 2)),
                sg.Button('Exit', size=(11, 2)),
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

window = sg.Window('Identikit generator', layout, finalize=True)

image = Image.open('./img_12382.png')
image.thumbnail((400, 400))
bio = io.BytesIO()
image.save(bio, format="PNG")
window["-IMAGE-"].update(data=bio.getvalue())

while True:  # Event Loop
    event, values = window.read()
    
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event == 'Show':
        # change the "output" element to be the value of "input" element
        window['-OUTPUT-'].update(values['-IN-'])
    else:
        event_parts = event.split('-')
        feature = event_parts[0]
        direction = event_parts[1]

        print(feature, direction)

window.close()