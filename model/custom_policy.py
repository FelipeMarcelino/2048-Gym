import tensorflow as tf
from stable_baselines.a2c.utils import conv, linear


def modified_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    # layer 1
    conv1 = conv(scaled_images, "c1", n_filters=128, filter_size=(1, 2), stride=1, init_scale=np.sqrt(2), **kwargs)
    conv2 = conv(scaled_images, "c2", n_filters=128, filter_size=(2, 1), stride=1, init_scale=np.sqrt(2), **kwargs)
    relu1 = activ(conv1)
    relu2 = activ(conv2)
    # layer 2
    conv11 = conv(relu1, "c3", n_filters=128, filter_size=(1, 2), stride=1, init_scale=np.sqrt(2), **kwargs)
    conv12 = conv(relu1, "c4", n_filters=128, filter_size=(2, 1), stride=1, init_scale=np.sqrt(2), **kwargs)
    conv21 = conv(relu2, "c3", n_filters=128, filter_size=(1, 2), stride=1, init_scale=np.sqrt(2), **kwargs)
    conv22 = conv(relu2, "c4", n_filters=128, filter_size=(2, 1), stride=1, init_scale=np.sqrt(2), **kwargs)
    # layer2 relu activation
    relu11 = tf.nn.relu(conv11)
    relu12 = tf.nn.relu(conv12)
    relu21 = tf.nn.relu(conv21)
    relu22 = tf.nn.relu(conv22)

    # get shapes of all activations
    shape1 = relu1.get_shape().as_list()
    shape2 = relu2.get_shape().as_list()

    shape11 = relu11.get_shape().as_list()
    shape12 = relu12.get_shape().as_list()
    shape21 = relu21.get_shape().as_list()
    shape22 = relu22.get_shape().as_list()

    # expansion
    hidden1 = tf.reshape(relu1, [-1, shape1[1] * shape1[2] * shape1[3]])
    hidden2 = tf.reshape(relu2, [-1, shape2[1] * shape2[2] * shape2[3]])

    hidden11 = tf.reshape(relu11, [-1, shape11[1] * shape11[2] * shape11[3]])
    hidden12 = tf.reshape(relu12, [-1, shape12[1] * shape12[2] * shape12[3]])
    hidden21 = tf.reshape(relu21, [-1, shape21[1] * shape21[2] * shape21[3]])
    hidden22 = tf.reshape(relu22, [-1, shape22[1] * shape22[2] * shape22[3]])

    # concatenation
    hidden = tf.concat([hidden1, hidden2, hidden11, hidden12, hidden21, hidden22], axis=1)

    # Linear layer 1
    linear_1 = activ(linear(hidden, scope="fc1", n_hidden=512, init_scale=np.sqrt(2)))

    # Linear layer 2
    linear_2 = activ(linear(linear_1, scope="fc2", n_hidden=128, init_scale=np.sqrt(2)))

    return linear_2

