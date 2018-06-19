import numpy as np
import scipy.misc
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

index = 0
for image in mnist.test.images:
    image = np.reshape(image, (28, 28))
    scipy.misc.imsave('images/%d.png' % index, image)
    index += 1
