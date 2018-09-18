import tensorflow as tf


def lrelu(x, leak=0.2, name='lrelu', alt_relu_impl=False):
    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak * x)


def instance_norm(x, epsilon = 1e-5):
    with tf.variable_scope('instance_norm'):
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable(
            'scale', [x.get_shape()[-1]],
            initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.02)
        )
        offset = tf.get_variable('offset', [x.get_sahpe()[-1]],
            initializer=tf.constant_initializer(0.0)
        )
        out = scale * tf.div(tf.subtract(x, mean), tf.sqrt(var + epsilon)) + offset
        return out


def general_conv2d(
    inputs, filters=64, 
    filter_height=7, filter_width=7, 
    stride_height=1, stride_width=1,
    stddev=0.02, padding='valid',
    name='conv2d', do_norm=True,
    do_relu=True, relufactor=0):

    with tf.variable_scope(name):
        conv = tf.layers.conv2d(
            inputs=inputs, 
            filters=filters, 
            kernel_size=[filter_height, filter_width], 
            strides=[stride_height, stride_width], 
            padding=padding,
            use_bias=True,
            activation=None, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev), 
            bias_initializer=tf.zeros_initializer()
        )
        if do_norm:
            conv = instance_norm(conv)

        if do_relu:
            if relufactor == 0:
                conv = tf.nn.relu(conv, name='relu')
            else:
                conv = lrelu(conv, relufactor, 'lrelu')

        return conv


def general_deconv2d(
    inputs, filters, 
    filter_height=7, filter_width=7, 
    stride_height=1, stride_width=1, 
    stddev=0.02, padding='valid', 
    name='deconv2d', do_norm=True, 
    do_relu=True, relufactor=0):

    with tf.variable_scope(name):
        conv = tf.layers.conv2d_transpose(
            inputs,
            filters,
            kernel_size=[filter_height, filter_width],
            strides=[stride_height, stride_width],
            padding=padding,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            bias_initializer=tf.zeros_initializer()
        )

        if do_norm:
            conv = instance_norm(conv)
        
        if do_relu:
            if relufactor == 0:
                conv = tf.nn.relu(conv, name='relu')
            else:
                conv = lrelu(conv, relufactor, name='lrelu')

        return conv


def resnet_block(input_res, filters):
    out_res_1 = general_conv2d(
        inputs=input_res,
        filters=filters,
        filter_width=3,
        filter_height=3,
        stride_width=1,
        stride_height=1,
    )

    out_res_2 = general_conv2d(
        inputs=out_res_1,
        filters=filters,
        filter_width=3,
        filter_height=3,
        stride_width=1,
        stride_height=1
    )

    return (out_res_2 + input_res)

