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

