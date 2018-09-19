import tensorflow as tf
from utils.helpers import get_weights, norm, leaky_relu, get_biases

## Discriminator layers
def Ck(input, k, slope=0.2, stride=2, reuse=False, norm='instance', is_training=True, name=None):
    """ A 4x4 Convolution-BatchNorm-LeakyReLU layer with k filters and stride 2
    Args:
        input: 4D tensor
        k: integer, number of filters (output depth)
        slope: LeakyReLU's slope
        stride: integer
        norm: 'instance' or 'batch' or None
        is_training: boolean or BoolTensor
        reuse: boolean
        name: string, e.g. 'C64'
    Returns:
        4D tensor
    """
    with tf.variable_scope(name, reuse=reuse):
        weights = get_weights('weights',
            shape=[4, 4, input.get_shape()[3], k])
        
        conv = tf.nn.conv2d(input, weights,
            strides=[1, stride, stride, 1], padding='same')
        normalized = norm(conv, is_training, norm)
        output = leaky_relu(normalized, slope)
        return output


def last_conv(input, reuse=False, use_sigmoid=False, name=None):
    """ Last convolutional layer of discriminator network
        (1 filter with size 4x4, stride 1)
    Args:
        input: 4D tensor
        reuse: boolean
        use_sigmoid: boolean (False if use lsgan)
        name: string, e.g. 'C64'
    """
    with tf.variable_scope(name, reuse=reuse):
        weights = get_weights('weights', 
            shape=[4, 4, input.get_shape()[3], 1])
        biases = get_biases('biases', [1])

        conv = tf.nn.conv2d(input, weights,
            strides=[1, 1, 1, 1], padding='same')
        output = conv + biases
        if use_sigmoid:
            output = tf.sigmoid(output)
        return output
        
        