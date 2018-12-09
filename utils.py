from keras.models import load_model

import matplotlib.pyplot as plt
import numpy as np

def grid_visual_mnist(data, gamma):
  """
  This function displays a grid of images to show full misclassification
  :param data: grid data of the form;
      [nb_classes : nb_classes : img_rows : img_cols : nb_channels]
  :return: if necessary, the matplot figure to reuse
  """
  import matplotlib.pyplot as plt

  # Ensure interactive mode is disabled and initialize our graph
  plt.ioff()
  figure = plt.figure()
  figure.canvas.set_window_title('MNIST Adversarial figure - Salency Map')

  figure.suptitle('Salency Map attack on MNIST gamma = {}'.format(gamma))

  # Add the images to the plot
  num_cols = data.shape[0]
  num_rows = data.shape[1]
  num_channels = data.shape[4]
  for y in range(num_rows):
    for x in range(num_cols):
      ax = figure.add_subplot(num_rows, num_cols, (x + 1) + (y * num_cols))
      if y == 4 and x ==0:
        ax.text(-0.5, 0.5, "Input Class",
          horizontalalignment='right',
          verticalalignment='center',
          rotation='vertical',
          transform=ax.transAxes)
      if x == 4 and y == 0: 
        ax.text(0.5, 1.5, "Output classification",
          horizontalalignment='center',
          verticalalignment='bottom',
          transform=ax.transAxes)
      if y == 0:
        ax.text(0.5, 1, str(x),
          horizontalalignment='center',
          verticalalignment='bottom',
          transform=ax.transAxes)
      if x == 0:
        ax.text(0, 0.5, str(y) ,
          horizontalalignment='right',
          verticalalignment='center',
          rotation='vertical',
          transform=ax.transAxes)
      plt.axis('off')
      if num_channels == 1:
        plt.imshow(data[x, y, :, :, 0], cmap='gray')
      else:
        plt.imshow(data[x, y, :, :, :])

  # Draw the plot and return
  plt.show()
  return figure

def plot_img(x):
    x = x.reshape([28, 28])
    plt.gray()
    plt.imshow(x)


# http://everettsprojects.com/2018/01/30/mnist-adversarial-examples.html
def stitch_images(images, y_img_count, x_img_count, margin = 2):
    
    # Dimensions of the images
    img_width = images[0].shape[0]
    img_height = images[0].shape[1]
    
    width = y_img_count * img_width + (y_img_count - 1) * margin
    height = x_img_count * img_height + (x_img_count - 1) * margin
    stitched_images = np.zeros((width, height, 3))

    # Fill the picture with our saved filters
    for i in range(y_img_count):
        for j in range(x_img_count):
            img = images[i * x_img_count + j]
            if len(img.shape) == 2:
                img = np.dstack([img] * 3)
            stitched_images[(img_width + margin) * i: (img_width + margin) * i + img_width,
                            (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    return stitched_images