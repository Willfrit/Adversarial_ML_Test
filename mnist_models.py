"""
This files from cleverhans tuto : https://arxiv.org/abs/1412.6572

Modified for IA Seminar (UCL - Rémy VOET)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import KerasModelWrapper, cnn_model
from cleverhans.utils_tf import model_eval
from keras import backend
from keras.models import load_model
from tensorflow.python.platform import flags

import utils

FLAGS = flags.FLAGS

FGMS_PARAMS = {'eps': 0.2, 'clip_min': 0., 'clip_max': 1.}

NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = .001
TRAIN_DIR = 'train_dir'
FILENAME = 'mnist.h5'

def load_tf():
    keras.layers.core.K.set_learning_phase(0)
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                        " to use the TensorFlow backend.")

    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
            "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)
    return report, sess


def mnist_normal(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                   learning_rate=LEARNING_RATE, train_dir=TRAIN_DIR,
                   filename=FILENAME,
                   testing=False, label_smoothing=0.1, defend=False):
    """
    MNIST CleverHans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param train_dir: Directory storing the saved model
    :param filename: Filename to save model under
    :param load_model: True for load, False for not load
    :param testing: if true, test error is calculated
    :param label_smoothing: float, amount of label smoothing for cross entropy
    :return: an AccuracyReport object
    """

    res_epoch_leg = []
    res_epoch_adv = []
    print("Load and set session of TF")
    report, sess = load_tf()

    print("Get MNIST data")
    mnist = MNIST(train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end)
    x_train, y_train = mnist.get_set('train')
    x_test, y_test = mnist.get_set('test')

    print("Define Model (Convolutional Neural Networks)")
    # Obtain Image Parameters
    img_rows, img_cols, nchannels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                            nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    model = cnn_model(img_rows=img_rows, img_cols=img_cols,
                        channels=nchannels, nb_filters=64,
                        nb_classes=nb_classes)

    wrap = KerasModelWrapper(model)
    preds = model(x)

    fgsm = FastGradientMethod(wrap, sess=sess)
    
    adv_x = fgsm.generate(x, **FGMS_PARAMS)
    # Consider the attack to be constant
    adv_x = tf.stop_gradient(adv_x)
    preds_adv = model(adv_x)

    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        eval_params = {'batch_size': batch_size}
        acc_leg = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
        report.clean_train_clean_eval = acc_leg
        # assert X_test.shape[0] == test_end - test_start, X_test.shape
        print('Test accuracy on legitimate examples: %0.4f' % acc_leg)
        res_epoch_leg.append([acc_leg])

        # Accuracy of the adversarially trained model on adversarial examples
        acc_adv = model_eval(sess, x, y, preds_adv, x_test, y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % acc_adv)
        report.adv_train_adv_eval = acc_adv
        res_epoch_adv.append([acc_adv])

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'filename': filename
    }

    print("Training from scratch.")
    att = None
    if defend:
        att = fgsm

    loss = CrossEntropy(wrap, smoothing=label_smoothing, attack=att)
    train(sess, loss, x_train, y_train, evaluate=evaluate, args=train_params)


    model.save(filename)

    return res_epoch_leg, res_epoch_adv

def main(argv=None):
    print("Begin MNIST Tuto")

    params = {
        'nb_epochs':FLAGS.nb_epochs,
        'batch_size':FLAGS.batch_size,
        'learning_rate':FLAGS.learning_rate,
        'train_dir':FLAGS.train_dir,
        'filename':"models/mnist_nor.h5", 
        'train_start':0,
        'train_end':600,
        'test_start':0,
        'test_end':100
    }

    res_l_n, res_a_n = mnist_normal(**params)

    params = {
        'nb_epochs':FLAGS.nb_epochs,
        'batch_size':FLAGS.batch_size,
        'learning_rate':FLAGS.learning_rate,
        'train_dir':FLAGS.train_dir,
        'filename':"models/mnist_def.h5", 
        'train_start':0,
        'train_end':600,
        'test_start':0,
        'test_end':100, 
        'defend': True
    }

    res_l_d, res_a_d = mnist_normal(**params)

    fig = plt.figure()
    ax = plt.axes()

    fig.suptitle("Accurate against legitimated test and adversarial test")

    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Accurate')

    ax.plot(range(NB_EPOCHS), res_l_n, label="Test - Normal CNN")
    ax.plot(range(NB_EPOCHS), res_a_n, label="Adversarial - Normal CNN")
    ax.plot(range(NB_EPOCHS), res_l_d, label="Test - Defend CNN")
    ax.plot(range(NB_EPOCHS), res_a_d, label="Adversarial - Defend CNN")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                        'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
    flags.DEFINE_float('learning_rate', LEARNING_RATE,
                        'Learning rate for training')
    flags.DEFINE_string('train_dir', TRAIN_DIR,
                        'Directory where to save model.')
    flags.DEFINE_string('filename', FILENAME, 'Checkpoint filename.')
    tf.app.run()
