import tensorflow as tf
from .resnet_block import resnet_block
from .convolutional import general_conv2d
from models.hlcgan_model import HLCGAN

def discriminator(input, name='discriminator'):
    with tf.variable_scope(name):
        f = 4 

        o_c1 = general_conv2d(input, HLCGAN.ngf, f, f, 2, 2, 0.02, 'same', 'c1', do_norm=False, relufactor=0.2)
        o_c2 = general_conv2d(o_c1, HLCGAN.ngf*2, f, f, 2, 2, 0.02, 'same', 'c2', relufactor=0.2)
        o_c3 = general_conv2d(o_c2, HLCGAN.ngf*4, f, f, 2, 2, 0.02, 'same', 'c3', relufactor=0.2)
        o_c4 = general_conv2d(o_c3, HLCGAN.ngf*8, f, f, 2, 2, 0.02, 'same', 'c4', relufactor=0.2)
        o_c5 = general_conv2d(o_c4, 1, f, f, 2, 2, 0.02, 'same', 'c5', do_norm=False, do_relu=False)

        return o_c5


def patch_discriminator(input, name='discriminator'):
    with tf.variable_scope(name):
        f = 4

        patch_input = tf.random_crop(input, [1, 70, 70, 3])
        o_c1 = general_conv2d(patch_input, HLCGAN.ngf, f, f, 2, 2, 0.02, 'same', 'c1', do_norm=False, relufactor=0.2)
        o_c2 = general_conv2d(o_c1, HLCGAN.ngf*2, f, f, 2, 2, 0.02, 'same', 'c2', relufactor=0.2)
        o_c3 = general_conv2d(o_c2, HLCGAN.ngf*4, f, f, 2, 2, 0.02, 'same', 'c3', relufactor=0.2)
        o_c4 = general_conv2d(o_c3, HLCGAN.ngf*8, f, f, 2, 2, 0.02, 'same', 'c4', relufactor=0.2)
        o_c5 = general_conv2d(o_c4, 1, f, f, 2, 2, 0.02, 'same', 'c5', do_norm=False, do_relu=False)

        return o_c5