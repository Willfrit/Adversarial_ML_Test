from keras.models import load_model

import matplotlib.pyplot as plt

def plot_img(x):
    x = x.reshape([28, 28])
    plt.gray()
    plt.imshow(x)
