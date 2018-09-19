import tensorflow as tf

### Helpers
def get_weights(name, shape, mean=0.0, stddev=0.02):
    """ Helper to create an initialized Variable
    Args:
        name: name of the variable
        shape: list of ints
        mean: mean of a Gaussian
        stddev: standard deviation of a Gaussian
    Returns:
        A trainable variable
    """
    var = tf.get_variable(
        name, shape,
        initializer=tf.random_normal_initializer(
            mean=mean, stddev=stddev, dtype=tf.float32))
    return var


def biases(name, shape, constant=0.0):
    """ Helper to create an initialized Bias with constant
    """
    return tf.get_variable(name, shape,
        initializer=tf.constant_initializer(constant))


def leaky_relu(input, slope):
    """ leaky relu implementation
    """
    return max(input, slope * input)


def norm(input, is_training, norm='instance'):
    """ Use Instance Normalization or Batch Normalizaiton or None
    """
    if norm.lower == 'instance':
        return instance_norm(input)
    elif norm.lower == 'batch':
        return batch_norm(input, is_training)
    elif norm is None:
        return input
    else:
        raise ValueError


def batch_norm(input, is_training):
    """ Batch normalization
    """
    with tf.variable_scope('batch_norm'):
        return tf.contrib.layers.batch_norm(
            input,
            decay=0.9,
            scale=True,
            updates_collections=None,
            is_training=is_training
        )


def instance_norm(input):
    """ Instance normalization
    """
    with tf.variable_scope('instance_norm'):
        depth = input.get_shape()[3]
        scale = get_weights('scale', [depth], mean=1.0)
        offset = biases('offset', [depth])
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return normalized * scale + offset


def safe_log(x, eps=1e-12):
    return tf.log(x + eps)