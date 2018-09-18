import tensorflow as tf
from .convolutional import general_conv2d, general_deconv2d
from models.hlcgan_model import HLCGAN

def resnet_block(inputs, dim, name='resnet'):
    with tf.variable_scope(name):
        out = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        out = general_conv2d(out, dim, 3, 3, 1, 1, 0.02, 
            padding='valid', name='c1'
        )
        out = tf.pad(out, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        out = general_conv2d(out, dim, 3, 3, 1, 1, 0.02, 
            padding='valid', name='c2', do_relu=False
        )

        return tf.nn.relu(out + inputs)


def generator_resnet_6blocks(inpus, name='generator'):
    with tf.variable_scope(name):
        f = 7
        ks = 3

        pad_input = tf.pad(input, [[0, 0], [ks, ks], [ks, ks], [0, 0]], 'REFLECT')
        o_c1 = general_conv2d(pad_input, HLCGAN.ngf, f, f, 1, 1, 0.02, name='c1')
        o_c2 = general_conv2d(o_c1, HLCGAN.ngf*2, ks, ks, 2, 2, 0.02, 'same', 'c2')
        o_c3 = general_conv2d(o_c2, HLCGAN.ngf*4, ks, ks, 2, 2, 0.02, 'same', 'c3')

        o_r1 = resnet_block(o_c3, HLCGAN.ngf*4, 'r1')
        o_r2 = resnet_block(o_r1, HLCGAN.ngf*4, 'r2')
        o_r3 = resnet_block(o_r2, HLCGAN.ngf*4, 'r3')
        o_r4 = resnet_block(o_r3, HLCGAN.ngf*4, 'r4')
        o_r5 = resnet_block(o_r4, HLCGAN.ngf*4, 'r5')
        o_r6 = resnet_block(o_r5, HLCGAN.ngf*4, 'r6')

        o_c4 = general_deconv2d(o_r6, HLCGAN.ngf*2, ks, ks, 2, 2, 0.02, 'same', 'c4')
        o_c5 = general_deconv2d(o_c4, HLCGAN.ngf, ks, ks, 2, 2, 0.02, 'same', 'c5')
        o_c5_pad = tf.pad(o_c5, [[0, 0], [ks, ks], [ks, ks], [0, 0]], 'REFLECT')
        o_c6 = general_conv2d(o_c5_pad, HLCGAN.img_layer, f, f, 1, 1, 0.02, 'valid', 'c6', do_relu=False)

        output = tf.nn.tanh(o_c6, name='tanh1')

        return output


def generator_resnet_9blocks(input, name='generator'):
    with tf.variable_scope(name):
        f = 7
        ks = 3

        pad_input = tf.pad(input, [[0, 0], [ks, ks], [ks, ks], [0, 0]], 'REFLECT')
        o_c1 = general_conv2d(pad_input, HLCGAN.ngf, f, f, 1, 1, 0.02, name='c1')
        o_c2 = general_conv2d(o_c1, HLCGAN.ngf*2, ks, ks, 2, 2, 0.02, 'same', 'c2')
        o_c3 = general_conv2d(o_c2, HLCGAN.ngf*4, ks, ks, 2, 2, 0.02, 'same', 'c3')

        o_r1 = resnet_block(o_c3, HLCGAN.ngf*4, 'r1')
        o_r2 = resnet_block(o_r1, HLCGAN.ngf*4, 'r2')
        o_r3 = resnet_block(o_r2, HLCGAN.ngf*4, 'r3')
        o_r4 = resnet_block(o_r3, HLCGAN.ngf*4, 'r4')
        o_r5 = resnet_block(o_r4, HLCGAN.ngf*4, 'r5')
        o_r6 = resnet_block(o_r5, HLCGAN.ngf*4, 'r6')
        o_r7 = resnet_block(o_r6, HLCGAN.ngf*4, 'r7')
        o_r8 = resnet_block(o_r7, HLCGAN.ngf*4, 'r8')
        o_r9 = resnet_block(o_r8, HLCGAN.ngf*4, 'r9')

        o_c4 = general_deconv2d(o_r9, HLCGAN.ngf*2, ks, ks, 2, 2, 0.02, 'same', 'c4')
        o_c5 = general_deconv2d(o_c4, HLCGAN.ngf, ks, ks, 2, 2, 0.02, 'same', 'c5')
        o_c6 = general_conv2d(o_c5, HLCGAN.img_layer, f, f, 1, 1, 0.02, 'same', 'c6', do_relu=False)

        output = tf.nn.tanh(o_c6, name='tanh1')

        return output


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
