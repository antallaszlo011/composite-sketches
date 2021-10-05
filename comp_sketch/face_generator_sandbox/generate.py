import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import PIL.Image

from tensorflow.python.client import device_lib

import time

# config = tf.ConfigProto(device_count = {'GPU': 1})
# sess = tf.Session(config=config)

OUT_DIR = './GEN_OUT/'

# Initialize TensorFlow session.
# sess = tf.InteractiveSession()
sess = tf.compat.v1.InteractiveSession()
print('Sess:', sess)

print('Initialized')

def read_axis(model='lin_reg', orto=False, norm=True):
    # path = '../DATA/feature_axis/' + model
    path = './'
    filename = 'feature_axis.csv'
    if orto and norm:
        filename = 'feature_axis_orthonorm.csv'
    elif norm:
        filename = 'feature_axis_norm.csv'
    # print('Reading:', os.path.join(path, filename))

    feature_axis = pd.read_csv(os.path.join(path, filename), index_col=0)
    # print(feature_axis.head())

    return feature_axis

def read_GAN():
    # Import official CelebA-HQ networks.
    with open('karras2018iclr-celebahq-1024x1024.pkl', 'rb') as file:
        G, D, Gs = pickle.load(file)

        return G, D, Gs

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


def generate_sample(G, D, Gs):
    h = 1.0
    features = ['Male', 'Eyeglasses', 'Smiling', 'Young', 'Wavy_Hair', 'Wearing_Necktie']
    print('Generating process start...')
    start_time = time.time()
    for k in range(75, 100):
        print('%d / 75 iteration completed' % k)
        print('Time elapsed: %s' % str(time.time() - start_time))

        curr_dir = os.path.join(OUT_DIR, "%03d" % k)
        # print(curr_dir)

        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)

        # Generate latent vectors.
        latent = np.random.randn(1, *Gs.input_shapes[0][1:]) # 1 random latent vector
        # print('Latent generated:', latent)

        # Get the corresponding image
        image = get_image_from_latent(latent, G, D, Gs)
        # print('Original image completed')

        # Save images as PNG.
        PIL.Image.fromarray(image, 'RGB').save(os.path.join(curr_dir, 'img_gen.png'))
        # print('Image saved')

        # -----------------------------------------------------------------------------

        feature_axis = read_axis(model='log_reg', orto=False, norm=True)
        # print('Feature axis read')

        basic_dir = os.path.join(curr_dir, 'basic')
        if not os.path.exists(basic_dir):
            os.makedirs(basic_dir)

        for feature in features:
            new_p_latent = np.copy(latent)
            new_n_latent = np.copy(latent)

            print('Current feature:', feature)
            for i in range(5):
                print('Step:', i+1, '/', 5)
                # Create new latent vectors
                new_p_latent = logistic_step(new_p_latent, feature_axis, feature=feature, h=+h)
                new_n_latent = logistic_step(new_n_latent, feature_axis, feature=feature, h=-h)

                # Get corresponding images
                p_image = get_image_from_latent(new_p_latent, G, D, Gs)
                n_image = get_image_from_latent(new_n_latent, G, D, Gs)

                # Save images as PNG.
                PIL.Image.fromarray(p_image, 'RGB').save(os.path.join(basic_dir, 'img_%s_p_%d_gen.png' % (feature, i)))
                PIL.Image.fromarray(n_image, 'RGB').save(os.path.join(basic_dir, 'img_%s_n_%d_gen.png' % (feature, i)))

        # -----------------------------------------------------------------------------

        feature_axis = read_axis(model='log_reg', orto=True, norm=True)
        # print('Feature axis read')

        orto_dir = os.path.join(curr_dir, 'orto')
        if not os.path.exists(orto_dir):
            os.makedirs(orto_dir)

        for feature in features:
            new_p_latent = np.copy(latent)
            new_n_latent = np.copy(latent)

            print('Current feature:', feature)
            for i in range(5):
                print('Step:', i+1, '/', 5)
                # Create new latent vectors
                new_p_latent = logistic_step(new_p_latent, feature_axis, feature=feature, h=+h)
                new_n_latent = logistic_step(new_n_latent, feature_axis, feature=feature, h=-h)

                # Get corresponding images
                p_image = get_image_from_latent(new_p_latent, G, D, Gs)
                n_image = get_image_from_latent(new_n_latent, G, D, Gs)

                # Save images as PNG.
                PIL.Image.fromarray(p_image, 'RGB').save(os.path.join(orto_dir, 'img_%s_p_%d_gen.png' % (feature, i)))
                PIL.Image.fromarray(n_image, 'RGB').save(os.path.join(orto_dir, 'img_%s_n_%d_gen.png' % (feature, i)))



if __name__=='__main__':
    tf.test.is_gpu_available()
    tf.test.gpu_device_name()

    print('Reading model...')
    G, D, Gs = read_GAN()
    print('Model read completed')
    generate_sample(G, D, Gs)