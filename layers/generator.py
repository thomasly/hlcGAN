import tensorflow as tf
from utils.helpers import get_weights, norm

## Layers: follow the naming convention used in the original paper
### Generator layers
def c7s1_k(input, k, reuse=False, norm='instance', activation='relu', is_training=True, name='c7s1_k'):
    """ A 7x7 Convolution-BatchNorm-ReLU layer with k filters and stride 1
    Args:
        input: 4D tensor
        k: integer, number of filters (output depth)
        norm: 'instance' or 'batch' or None
        activation: 'relu' or 'tanh'
        name: string, e.g. 'c7sk-32'
        is_training: boolean or BoolTensor
        name: string
        reuse: boolean
    Returns:
        4D tensor
    """
    with tf.variable_scope(name, reuse=reuse):
        weights = get_weights("weights",
            shape=[7, 7, input.get_shape()[3], k])
        
        padded = tf.pad(input, [[0, 0], [3, 3], [3, 3], [0,0]], 'REFLECT')
        conv = tf.nn.conv2d(padded, weights, 
            strides=[1, 1, 1, 1], padding='valid')

        normalized = norm(conv, is_training, norm)
        if activation == 'relu':
            output = tf.nn.relu(normalized)
        elif activation == 'tanh':
            output = tf.nn.tanh(normalized)
        else:
            raise ValueError
        return output


def dk(input, k, reuse=False, norm='instance', is_training=True, name=None):
    """ A 3x3 Convolution-BatchNorm-ReLU layer with k filters and stride 2
    Args:
        input: 4D tensor
        k: integer, number of filters (output depth)
        norm: 'instance' or 'batch' or None
        is_training: boolean or BoolTensor
        name: string
        reuse: boolean
        name: string, e.g. 'd64'
    Returns:
        4D tensor
    """
    with tf.variable_scope(name, reuse=reuse):
        weights = get_weights('weights',
            shape=[3, 3, input.get_shape()[3], k])
        
        conv = tf.nn.conv2d(input, weights, 
            strides=[1, 2, 2, 1], padding='same')
        normalized = norm(conv, is_training, norm)
        output = tf.nn.relu(normalized)
        return output


def Rk(input, k, reuse=False, norm='instance', is_training=True, name=None):
    """ A residual block that contains two 3x3 convolutional layers
        with the same number of filters on both layer
        Args:
            input: 4D Tensor
            k: integer, number of filters (output depth)
            reuse: boolean
            name: string
        Returns:
            4D tensor (same shape as input)
    """
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('layer1', reuse=reuse):
            weights1 = get_weights('weights1',
                shape=[3, 3, input.get_shape()[3], k])
            padded1 = tf.pad(input, [[0,0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            conv1 = tf.nn.conv2d(padded1, weights1,
                strides=[1, 1, 1, 1], padding='valid')
            normalized1 = norm(conv1, is_training, norm)
            relu1 = tf.nn.relu(normalized1)

        with tf.variable_scope('layer2', reuse=reuse):
            weights2 = get_weights('weights2',
                shape=[3, 3, relu1.get_shape()[3], k])
            padded2 = tf.pad(relu1, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            conv2 = tf.nn.conv2d(padded2, weights2,
                strides=[1, 1, 1, 1], padding='valid')
            normalized2 = norm(conv2, is_training, norm)
        
        output = tf.add(normalized2, input)
        return output
        
        
def n_res_block(input, reuse=False, norm='instance', is_training=True, n=6):
    """ resNet blocks
    Args:
        input: 4D tensor
        reuse: boolean
        norm: string, 'instance', 'batch' or None
        is_training: boolean
        n: integer, number of the reNet blocks
    Returns:
        4D tensor (same shape as input)
    """
    depth = input.get_shape()[3]
    for i in range(1, n+1):
        output = Rk(input, depth, reuse, norm, is_training, 'R{}_{}'.format(depth, i))
        input = output
    return output


def uk(input, k, reuse=False, norm='instance', is_training=True, name=None, output_size=None):
    """ A 3x3 fractional-strided-Convolution-BatchNorm-ReLU layer
        with k filters, stride 1/2
    Args:
        input: 4D tensor
        k: integer, number of filters (output depth)
        norm: 'instance' or 'batch' or None
        is_training: boolean or BoolTensor
        reuse: boolean
        name: string, e.g. 'c7sk-32'
        output_size: integer, desired output size of layer
    Returns:
        4D tensor
    """
    with tf.variable_scope(name, reuse=reuse):
        input_shape = input.get_shape().as_list()

        weights = get_weights('weights',
            shape=[3, 3, k, input_shape[3]])
        
        if not output_size:
            output_size = input_shape[1] * 2
        output_shape = [input_shape[0], output_size, output_size, k]
        fsconv = tf.nn.conv2d_transpose(input, weights,
            output_shape=output_shape,
            strides=[1, 2, 2, 1], padding='same')
        normalized = norm(fsconv, is_training, norm)
        output = tf.nn.relu(normalized)
        return output


