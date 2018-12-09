"""
From : http://everettsprojects.com/2018/01/30/mnist-adversarial-examples.html
modified/simplified 
"""
from sklearn.model_selection import train_test_split
import pandas as pd

import numpy as np
import keras
from keras import backend
from keras.models import load_model
import tensorflow as tf

from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.dataset import MNIST
from cleverhans.utils_tf import model_eval


from matplotlib import pyplot as plt
import imageio

from utils import stitch_images

# Set the matplotlib figure size
plt.rc('figure', figsize = (12.0, 12.0))

# Set the learning phase to false, the model is pre-trained.
backend.set_learning_phase(False)
keras_model = load_model('models/mnist_def.h5')
# FOR LAUNCH ON DEFEND model
# keras_model = load_model('models/mnist_def.h5')


'''
Split the provided training data to create a new training
data set and a new validation data set. These will be used
or hyper-parameter tuning.
'''
# Use the same seed to get the same validation set
seed = 27

mnist = MNIST(test_start=0, test_end=1000)
x_validation, y_validation = mnist.get_set('test')

# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

if not hasattr(backend, "tf"):
    raise RuntimeError("This tutorial requires keras to be configured"
                       " to use the TensorFlow backend.")

if keras.backend.image_dim_ordering() != 'tf':
    keras.backend.set_image_dim_ordering('tf')
    print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
          "'th', temporarily setting to 'tf'")

# Retrieve the tensorflow session
sess =  backend.get_session()

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y = tf.placeholder(tf.float32, shape=(None, 10))

preds = keras_model(x)

eval_params = {'batch_size': 128}
acc = model_eval(sess, x, y, preds, x_validation, y_validation, args=eval_params)

print("The normal validation accuracy is: {}".format(acc))

# Initialize the Fast Gradient Sign Method (FGSM) attack object and 
# use it to create adversarial examples as numpy arrays.
wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.20,
               'clip_min': 0.,
               'clip_max': 1.}

adv_x = fgsm.generate(x, **fgsm_params)
# Consider the attack to be constant
adv_x = tf.stop_gradient(adv_x)
preds_adv = keras_model(adv_x)

adv_acc = model_eval(sess, x, y, preds_adv, x_validation, y_validation, args=eval_params)

print("The adversarial validation accuracy is: {}".format(adv_acc))


adv_x_np = fgsm.generate_np(x_validation[:10], **fgsm_params)

x_sample = x_validation[0].reshape(28, 28)
adv_x_sample = adv_x_np[0].reshape(28, 28)

adv_comparison = stitch_images([x_sample, adv_x_sample], 1, 2)

## PREDICTED CLASS
normal_digit_img = x_sample.reshape(1, 28, 28, 1)
adv_digit_img = adv_x_sample.reshape(1, 28, 28, 1)

normal_digit_pred = np.argmax(keras_model.predict(normal_digit_img), axis = 1)
adv_digit_pred = np.argmax(keras_model.predict(adv_digit_img), axis = 1)

print('The normal digit is predicted to be a {}'.format(normal_digit_pred))
print('The adversarial example digit is predicted to be an {}'.format(adv_digit_pred))

fig = plt.figure()
ax = plt.axes()

fig.suptitle('{} tagerted as {}'.format(normal_digit_pred, adv_digit_pred))

plt.imshow(adv_comparison)
plt.show()