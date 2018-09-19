import tensorflow as tf
from layers import generator_layers as gl
from utils import helpers

class Generator:
    def __init__(self, name, is_training, ngf=64, 
        norm='instance', image_size=128):

        self.name = name
        self.is_training = is_training
        self.reuse = False
        self.ngf = ngf
        self.norm = norm
        self.image_size = image_size


    def __call__(self, input):
        """
        Args:
        input: batch_size x width x height x 3
        Returns:
        output: same size as input
        """
        with tf.variable_scope(self.name):
            # conv layers
            c7s1_32 = gl.c7s1_k(input, self.ngf, 
                is_training=self.is_training, norm=self.norm,
                reuse=self.reuse, name='c7s1_32')  # (?, w, h, 32)
            d64 = gl.dk(c7s1_32, 2*self.ngf, is_training=self.is_training,
                norm=self.norm, reuse=self.reuse, name='d64')  # (?, w/2, h/2, 64)
            d128 = gl.dk(d64, 4*self.ngf, is_training=self.is_training,
                norm=self.norm, reuse=self.reuse, name='d128')  # (?, w/4, h/4, 128)

            if self.image_size <= 128:
                # use 6 residual blocks for 128x128 images
                res_out = gl.n_res_block(d128, reuse=self.reuse, n=6)  # (?, w/4, w/4, 128)
            else:
                # 9 blocks for higher resolution
                res_out = gl.n_res_block(d128, reuse=self.reuse, n=9)  # (?, w/4, w/4, 128)

            # factional-strided convolution
            u64 = gl.uk(res_out, 2*self.ngf, reuse=self.reuse, norm=self.norm,
                is_training=self.is_training, name = 'u64')  # (?, w/2, h/2, 64)
            u32 = gl.uk(u64, self.ngf, reuse=self.reuse, norm=self.norm,
                is_training=self.is_training, name='u32')  # (?, w, h, 32)

            # conv layer
            output = gl.c7s1_k(u32, 3, norm=None,
                activation='tanh', reuse=self.reuse, name='output')  # (?, w, h, 3)
            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output

            
    def sample(self, input):
        image = helpers.batch_convert2int(self.__call__(input))
        image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
        return image
            