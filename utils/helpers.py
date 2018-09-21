import tensorflow as tf
import random


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


def get_biases(name, shape, constant=0.0):
    """ Helper to create an initialized Bias with constant
    """
    return tf.get_variable(
        name, shape,
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
        offset = get_biases('offset', [depth])
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return normalized * scale + offset


def safe_log(x, eps=1e-12):
    return tf.log(x + eps)


def convert2int(image):
    """ Transfrom from float tensor ([-1.,1.]) to int image ([0,255])
    """
    return tf.image.convert_image_dtype(
        (image+1.0)/2.0,
        tf.uint8, saturate=True)


def convert2float(image):
    """ Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return (image / 127.5) - 1.0


def batch_convert2int(images):
    """
    Args:
        images: 4D float tensor (batch_size, image_size, image_size, depth)
    Returns:
        4D int tensor
    """
    return tf.map_fn(convert2int, images, dtype=tf.uint8)


def batch_convert2float(images):
    """
    Args:
        images: 4D int tensor (batch_size, image_size, image_size, depth)
    Returns:
        4D float tensor
    """
    return tf.map_fn(convert2float, images, dtype=tf.float32)


class ImagePool:
    """ History of generated images
        Same logic as
        https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, image):
        if self.pool_size == 0:
            return image

        if len(self.images) < self.pool_size:
            self.images.append(image)
            return image
        else:
            p = random.random()
            if p > 0.5:
                # use old image
                random_id = random.randrange(0, self.pool_size)
                tmp = self.images[random_id].copy()
                self.images[random_id] = image.copy()
                return tmp
            else:
                return image
