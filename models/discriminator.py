import tensorflow as tf 
from layers import discriminator_layers as dl

class Discriminator:
    def __init__(self, name, is_training, norm='instance', use_sigmoid=False):
        self.name = name
        self.is_training = is_training
        self.norm = norm
        self.reuse = False
        self.use_sigmoid = use_sigmoid

    def __call__(self, input):
        """
        Args:
        input: batch_size x image_size x image_size x 3
        Returns:
        output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
                filled with 0.9 if real, 0.0 if fake
        """
        with tf.variable_scope(self.name):
            # convolutional layers
            C64 = dl.Ck(input, 64, reuse=self.reuse, norm=None,
                is_training=self.is_training, name='C64')   # (?, w/2, h/2, 64)
            C128 = dl.Ck(C64, 128, reuse=self.reuse, norm=self.norm,
                is_training=self.is_training, name='C128')  # (?, w/4, h/4, 128)
            C256 = dl.Ck(C128, 256, reuse=self.reuse, norm=self.norm,
                is_training=self.is_training, name='C256')  # (?, w/8, h/8, 256)
            C512 = dl.Ck(C256, 512, reuse=self.reuse, norm=self.norm,
                is_training=self.is_training, name='C512')  # (?, w/16, h/16, 512)

            # apply a convolution to produce a 1 dimensional output
            # use sigmoid=False if use_lsgan=True
            output = dl.last_conv(C512, reuse=self.reuse,
                use_sigmoid=self.use_sigmoid, name='output') # (?, w/16, h/16, 1)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output
        