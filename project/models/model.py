import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def cnn_model():
    X = tf.placeholder(tf.float32, [None, 28 * 28])
    y = tf.placeholder(tf.float32, [None, 10])

    # input data, a 4-D tensor
    input = tf.reshape(X, [-1, 28, 28, 1])

    # convolutional layer 1
    convo1 = tf.layers.conv2d(input, 32, 5, activation=tf.nn.relu)
    # pooling layer 1
    pool1 = tf.layers.max_pooling2d(convo1, 2, 2)
    # convolutional layer 2
    convo2 = tf.layers.conv2d(input, 64, 5, activation=tf.nn.relu)
    # pooling layer 2
    pool2 = tf.layers.max_pooling2d(convo2, 2, 2)

    # reshape to 2-D
    # dense_input = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense_input = tf.contrib.layers.flatten(pool2)

    # add a dense layer
    dense1 = tf.layers.dense(dense_input, 1024)
    # add dropout
    prob = tf.placeholder_with_default(1.0, shape=())
    dropout = tf.layers.dropout(
        inputs=dense1,
        rate=prob,
    )
    # add final logit layer
    logits = tf.layers.dense(
        inputs=dropout,
        units=10,
    )

    # add cost function
    cost = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=logits,
    )
    # define the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
    # define the train op
    train_op = optimizer.minimize(
        loss=cost,
        global_step=tf.train.get_global_step()
    )
    # calculate predictions
    return [X, y, train_op, cost, logits, prob]


def dense_model():
    X = tf.placeholder(tf.float32, [None, 28 * 28])
    y = tf.placeholder(tf.float32, [None, 10])

    # dense layer 1
    dense1 = tf.layers.dense(
        inputs=X,
        units=1024,
        activation=tf.nn.relu,
    )
    # dense layer 2
    dense2 = tf.layers.dense(
        inputs=dense1,
        units=1024,
        activation=tf.nn.relu,
    )
    # add dropout
    dropout = tf.layers.dropout(
        inputs=dense1,
        rate=0.4,
    )
    # add final logistic regression layer
    output = tf.layers.dense(
        inputs=dense2,
        units=10,
        activation=tf.nn.relu,
    )

    # add cost function
    cost = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=output,
    )
    # define the optimizer
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(0.6, global_step, 10000, 0.96, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3)
    # define the train op
    train_op = optimizer.minimize(
        loss=cost,
        global_step=tf.train.get_global_step()
    )
    return [X, y, train_op, cost, output]


def eval(onehot_preds, onehot_labels):
    preds = np.argmax(onehot_preds, axis=1)
    labels = np.argmax(onehot_labels, axis=1)
    assert len(preds) == len(labels)
    count = 0
    for i in range(len(preds)):
        if preds[i] == labels[i]:
            count += 1
    return count * 1.0 / len(preds)


def test():
    pass


def train():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    step = 5000
    batch_size = 128
    eval_size = 100

    with tf.Session() as sess:
        # X, y, train_op, cost, output = dense_model()
        X, y, train_op, cost, output, prob = cnn_model()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for i in range(step):
            batch_X, batch_y = mnist.train.next_batch(batch_size)
            # print cost.eval(feed_dict={X: batch_X, y: batch_y})
            train_op.run(feed_dict={X: batch_X, y: batch_y, prob: 0.25})
            if i % eval_size == 0:
                onehot_preds = output.eval(feed_dict={X: mnist.validation.images})
                acc = eval(onehot_preds, mnist.validation.labels)
                print 'acc: ', acc
                if acc > .988:
                    break

        # test acc
        onehot_preds = output.eval(feed_dict={X: mnist.test.images})
        print 'test acc: ', eval(onehot_preds, mnist.test.labels)

        # save model
        saver = tf.train.Saver()
        saver.save(sess, "model/model.ckpt")


def predict(images):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    images = np.reshape(images, (28 * 28))
    with tf.Session() as sess:
        X, y, train_op, cost, output, prob = cnn_model()
        # restore model
        saver = tf.train.Saver()
        saver.restore(sess, "model/model.ckpt")
        onehot_preds = output.eval(feed_dict={X: images})
        print 'test acc: ', eval(onehot_preds, mnist.test.labels)
        preds = np.argmax(onehot_preds, axis=1)
        return preds


if __name__ == '__main__':
    train()
