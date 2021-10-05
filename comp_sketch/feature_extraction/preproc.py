import pandas as pd
import numpy as np
import imutils
import glob
import cv2
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

# pd.set_option('display.max_columns', 500)
tqdm().pandas()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(origin, destination, fc):
    ## Importing image using open cv
    img = cv2.imread(origin,1)

    ## Resizing to constant width
    img = imutils.resize(img, width=200)
    
    ## Finding actual size of image
    H,W,_ = img.shape
    
    ## Converting BGR to RGB
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ## Detecting faces on the image
    face_coord = fc.detectMultiScale(gray,1.2,10,minSize=(50,50))
    
    ## If only one face is foung
    if len(face_coord) == 1:
        X, Y, w, h = face_coord[0]
    
    ## If no face found --> SKIP
    elif len(face_coord)==0:
        return None
    
    ## If multiple faces are found take the one with largest area
    else:
        max_val = 0
        max_idx = 0
        for idx in range(len(face_coord)):
            _, _, w_i, h_i = face_coord[idx]
            if w_i*h_i > max_val:
                max_idx = idx
                max_val = w_i*h_i
            else:
                pass
            
            X, Y, w, h = face_coord[max_idx]
    
    ## Crop and export the image
    img_cp = img[
            max(0,Y - int(0.35*h)): min(Y + int(1.35*h), H),
            max(0,X - int(w*0.35)): min(X + int(1.35*w), W)
        ].copy()
    
    # print(destination)
    cv2.imwrite(destination, img_cp)

## Defining destination path
path = 'DATA\\faces\\'

# Finding all the images in the folder
item_list = glob.glob('DATA\\img_align_celeba\\*.jpg')
print(len(item_list))

# Will run for about an hour and a half 
for org in tqdm(item_list):
    print(org, path+org.split('\\')[-1])
    face_extractor(origin = org, destination = path+org.split('\\')[-1], fc=face_cascade)

# Findign all the images and separating in training and validation
item_list = glob.glob(path+'*.jpg')

for idx in tqdm(range(1,202600)):
    if idx <= 182637:
        destination = path+'training\\'
    else:
        destination = path+'validation\\'
    try:
        # print(path+str(idx).zfill(6)+'.jpg', destination+str(idx).zfill(6)+'.jpg')
        shutil.move(
            path+str(idx).zfill(6)+'.jpg', 
            destination+str(idx).zfill(6)+'.jpg'
        )
    except:
        pass

Combining all label attributes
label_df = pd.read_csv('DATA\\list_attr_celeba.csv')
column_list = pd.Series(list(label_df.columns)[1:])

def label_generator(row):
    return(' '.join(column_list[[True if i==1 else False for i in row[column_list]]]))

label_df['label'] = label_df.progress_apply(lambda x: label_generator(x), axis=1)
label_df = label_df.loc[:,['image_id','label']]
label_df.to_csv('DATA\\labels.csv')

## Attachhing label to correct file names
item_list = glob.glob('DATA\\faces\\*\\*.jpg')
item_df = pd.DataFrame({'image_name': pd.Series(item_list).progress_apply(lambda x: '\\'.join(x.split('\\')[-2:]))})
item_df['image_id'] = item_df.image_name.progress_apply(lambda x: x.split('\\')[1])

## Creating final label set
label_df = pd.read_csv('DATA\\labels.csv')
label_df = label_df.merge(item_df, on='image_id', how='inner')
label_df.rename(columns={'label':'tags'}, inplace=True)
label_df.loc[:,['image_name','tags']].to_csv('DATA\\faces\\labels.csv', index=False)

