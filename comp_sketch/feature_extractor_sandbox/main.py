import numpy as np
import pandas as pd
import cv2
import imutils
import json
import os
from pathlib import Path
from fastai.vision.data import ImageList
from fastai.vision.learner import cnn_learner
from fastai.vision import models
from fastai.vision.image import pil2tensor,Image


print('Import completed')

path = Path("./DATA/")

SELECTED_FEATURES = ['Eyeglasses', 'Male', 'Mustache', 'Smiling', 'Wavy_Hair', 'Young']

statistics = {feature: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for feature in SELECTED_FEATURES}

# Creating a databunch
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
data = (
    ImageList.from_csv(path, csv_name="labels.csv")
    .split_none()
    .label_from_df(label_delim=" ")
    .transform(None, size=128)
    .databunch(no_check=True)
    .normalize(imagenet_stats)
)

print('Data ready')

# Loading our model
learn = cnn_learner(data, models.resnet50, pretrained=False)
learn.load("ff_stage-2-rn50")

print('CNN load')

# Loading HAAR cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

print('Haar completed')

def detect_facial_attributes(frame):
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find faces using Haar cascade
    face_coord = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    max_area = 0
    max_X = 0
    max_Y = 0
    max_w = 0
    max_h = 0

    ## Looping through each face
    for coords in face_coord:
        ## Finding co-ordinates of face
        X, Y, w, h = coords

        area = w * h
        if area > max_area:
            max_area = area
            max_X = X
            max_Y = Y
            max_w = w
            max_h = h

    if max_area == 0:
        return None

    ## Finding frame size
    H, W, _ = frame.shape

    ## Computing larger face co-ordinates
    X_1, X_2 = (max(0, max_X - int(max_w * 0.35)), min(max_X + int(1.35 * max_w), W))
    Y_1, Y_2 = (max(0, max_Y - int(0.35 * max_h)), min(max_Y + int(1.35 * max_h), H))

    ## Cropping face and changing BGR To RGB
    img_cp = frame[Y_1:Y_2, X_1:X_2].copy()
    img_cp1 = cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB)

    # cv2.imshow('original', frame)
    # cv2.imshow('haar', img_cp)
    # cv2.imshow('color', img_cp1)
    # cv2.waitKey()

    ## Prediction of facial featues
    prediction = str(
        learn.predict(Image(pil2tensor(img_cp1, np.float32).div_(255)))[0]
    ).split(";")
    
    return set(prediction)


if __name__ == "__main__":
      
    
#     for filename in os.listdir(IMAGE_DIRECTORY):
#         if filename.endswith(".png"):
#             latents = np.load(latents_file)
#             img_name = imgs_file.readline()[:-1]
#             # print(latents)
#             # print(latents.shape)
#             # print(img_name)
#             # print()
#             # print(os.path.join(IMAGE_DIRECTORY, filename))
#             img = cv2.imread(os.path.join(IMAGE_DIRECTORY, filename))
#             prediction = detect_facial_attributes(img)

#             if prediction != None:
#                 img_desc = dict()
#                 img_desc['name'] = img_name
#                 img_desc['latent_vect'] = latents
#                 img_desc['features'] = prediction

#                 print(img_name)
#                 np.save(out_file, img_desc)            

    # img = cv2.imread('original.png')
    # prediction = detect_facial_attributes(img)
    # print(prediction)

    df = pd.read_csv('./DATA/labels.csv')
    for index, row in df.iterrows():
        print(index, '/', len(df))
        img_name = row['image_name']
        tags = set(row['tags'].split(' '))
        img = cv2.imread(os.path.join('./DATA/', img_name))
        prediction = detect_facial_attributes(img)

        if prediction == None:
            continue

        for feature in SELECTED_FEATURES:
            if feature in tags:
                if feature in prediction:
                    statistics[feature]['TP'] = statistics[feature]['TP'] + 1
                else:
                    statistics[feature]['FN'] = statistics[feature]['FN'] + 1
            else:
                if feature in prediction:
                    statistics[feature]['FP'] = statistics[feature]['FP'] + 1
                else:
                    statistics[feature]['TN'] = statistics[feature]['TN'] + 1

    print(statistics)
    with open('final_statistics.json', 'w') as f:
        json.dump(statistics, f)

        
