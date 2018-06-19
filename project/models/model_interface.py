import tensorflow as tf

import numpy as np
from model import cnn_model


def singleton(cls):
    instance = cls()
    instance.__call__ = lambda: instance
    return instance


@singleton
class ModelInterface(object):
    def __init__(self):
        self.sess = tf.Session()
        self.X, self.y, self.train_op, self.cost, self.output, self.prob = cnn_model()
        # restore model
        saver = tf.train.Saver()
        saver.restore(self.sess, "project/models/model/model.ckpt")

    def predict(self, images):
        images = np.reshape(images, (-1, 28 * 28))
        onehot_preds = self.output.eval(session=self.sess, feed_dict={self.X: images})
        preds = np.argmax(onehot_preds, axis=1)
        return preds
