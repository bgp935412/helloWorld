import tensorflow as tf
from tensorflow import contrib as tf_contrib
##################################################################################
# Layer
##################################################################################
DEFAULT_PADDING='SAME'
weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.05)
weight_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv'):
    with tf.variable_scope(scope):
        if pad > 0:
            if (kernel - stride) % 2 == 0:
                pad_top = pad
                pad_bottom = pad
                pad_left = pad
                pad_right = pad

            else:
                pad_top = pad
                pad_bottom = kernel - stride - pad_top
                pad_left = pad
                pad_right = kernel - stride - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x


def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=False, scope='deconv'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()

        if padding == 'SAME':
            output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]

        else:
            output_shape = [x_shape[0], x_shape[1] * stride + max(kernel - stride, 0),
                            x_shape[2] * stride + max(kernel - stride, 0), channels]

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape,
                                       strides=[1, stride, stride, 1], padding=padding)

            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init,
                                           kernel_regularizer=weight_regularizer,
                                           strides=stride, padding=padding, use_bias=use_bias)

        return x


def fully_conneted(x, channels, use_bias=True, sn=False, scope='fully'):
    with tf.variable_scope(scope):
        x = tf.layers.flatten(x)
        shape = x.get_shape().as_list()
        x_channel = shape[-1]

        if sn:
            w = tf.get_variable("kernel", [x_channel, channels], tf.float32, initializer=weight_init,
                                regularizer=weight_regularizer)
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else:
            x = tf.layers.dense(x, units=channels, kernel_initializer=weight_init,
                                kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x


def gaussian_noise_layer(x, is_training=False):
    if is_training:
        noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)
        return x + noise

    else:
        return x


def validate_padding(padding):
    assert padding in ('SAME', 'VALID')


def max_pool(input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
    validate_padding(padding)
    return tf.nn.max_pool(input,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)


def make_var(name, shape, initializer=None, trainable=True):
    x = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return x


def conv_pretrain(input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1, trainable=True):
    validate_padding(padding)
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        #            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.1)
        init_weights = tf.truncated_normal_initializer(stddev=0.02)
        init_biases = tf.constant_initializer(0.0)
        kernel = make_var('weights', [k_h, k_w, int(c_i) / group, c_o], init_weights, trainable)
        biases = make_var('biases', [c_o], init_biases, trainable)
        if group == 1:
            conv = convolve(input, kernel)
        else:
            input_groups = tf.split(3, group, input)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            conv = tf.concat(3, output_groups)
        if relu:
            bias = tf.nn.bias_add(conv, biases)
            return tf.nn.relu(bias, name=scope.name)
        return tf.nn.bias_add(conv, biases, name=scope.name)


def fully_conneted_pretrain(input, num_in, num_out, name, Activation=None, padding=DEFAULT_PADDING, trainable=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        init_weights = tf.truncated_normal_initializer(0.0, stddev=0.1)
        init_biases = tf.constant_initializer(0.0)
        weights = make_var('weights', [num_in, num_out], init_weights, trainable)
        biases = make_var('biases', [num_out], init_biases, trainable)
        flat = tf.reshape(input, [-1, weights.get_shape().as_list()[0]])
        act = tf.nn.xw_plus_b(flat, weights, biases, name=name)
        if Activation == 'relu':
            relu = tf.nn.relu(act)
            return relu
        elif Activation == 'sigmoid':
            sigmoid = tf.nn.sigmoid(act)
            return sigmoid
        elif Activation == 'tanh':
            tanh = tf.nn.tanh(act)
            return tanh
        elif Activation == 'sofmax':
            sofmax = tf.nn.sofmax(act)
            return sofmax
        elif Activation == None:
            return act


##################################################################################
# Block
##################################################################################
def flow_masked_conv(dx,dy,channel, kernel, stride, pad, scope, pad_type='reflect'):
    with tf.variable_scope('FMC_'+scope, reuse = tf.AUTO_REUSE):
        preactivate_dx_ = conv(dx, channel, kernel, stride, pad, pad_type, scope+'conv_dx')
        preactivate_dy_ = conv(dy, channel, kernel, stride, pad, pad_type, scope+'conv_dy')
        dx_ = tf.nn.tanh(preactivate_dx_)
        dy_ = tf.nn.tanh(preactivate_dy_)
        r_ = tf.sqrt(tf.square(preactivate_dx_) + tf.square(preactivate_dy_))
        r = tf.layers.dense(r_,channel,activation=tf.nn.sigmoid)
        return r*dx_, r*dy_


def flow_masked_resblock(dx_init ,dy_init , channels, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            dx, dy = flow_masked_conv(dx_init, dy_init, channels, kernel=3, stride=1, pad=1, scope = 'conv1', pad_type='reflect')
            dx, dy = tf.split(instance_norm(tf.concat([dx,dy],axis = -1)),2,axis = -1)

        with tf.variable_scope('res2'):
            dx, dy = flow_masked_conv(dx, dy, channels, kernel=3, stride=1, pad=1, scope = 'conv2', pad_type='reflect')
            dx, dy = tf.split(instance_norm(tf.concat([dx,dy],axis = -1)),2,axis = -1)

        return dx + dx_init, dy + dy_init


def resblock(x_init, channels, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = instance_norm(x)

        return x + x_init


def basic_block(x_init, channels, use_bias=True, sn=False, scope='basic_block'):
    with tf.variable_scope(scope):
        x = lrelu(x_init, 0.2)
        x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

        x = lrelu(x, 0.2)
        x = conv_avg(x, channels, use_bias=use_bias, sn=sn)

        shortcut = avg_conv(x_init, channels, use_bias=use_bias, sn=sn)

        return x + shortcut


def mis_resblock(x_init, z, channels, use_bias=True, sn=False, scope='mis_resblock'):
    with tf.variable_scope(scope):
        z = tf.reshape(z, shape=[-1, 1, 1, z.shape[-1]])
        z = tf.tile(z, multiples=[1, x_init.shape[1], x_init.shape[2], 1])  # expand

        with tf.variable_scope('mis1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn,
                     scope='conv3x3')
            x = instance_norm(x)

            x = tf.concat([x, z], axis=-1)
            x = conv(x, channels * 2, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_0')
            x = relu(x)

            x = conv(x, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_1')
            x = relu(x)

        with tf.variable_scope('mis2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn,
                     scope='conv3x3')
            x = instance_norm(x)

            x = tf.concat([x, z], axis=-1)
            x = conv(x, channels * 2, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_0')
            x = relu(x)

            x = conv(x, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_1')
            x = relu(x)

        return x + x_init


def avg_conv(x, channels, use_bias=True, sn=False, scope='avg_conv'):
    with tf.variable_scope(scope):
        x = avg_pooling(x, kernel=2, stride=2)
        x = conv(x, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn)

        return x


def conv_avg(x, channels, use_bias=True, sn=False, scope='conv_avg'):
    with tf.variable_scope(scope):
        x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
        x = avg_pooling(x, kernel=2, stride=2)

        return x


def expand_concat(x, z):
    z = tf.reshape(z, shape=[z.shape[0], 1, 1, -1])
    z = tf.tile(z, multiples=[1, x.shape[1], x.shape[2], 1])  # expand
    x = tf.concat([x, z], axis=-1)

    return x


##################################################################################
# Sampling
##################################################################################

def down_sample(x):
    return avg_pooling(x, kernel=3, stride=2, pad=1)


def avg_pooling(x, kernel=2, stride=2, pad=0):
    if pad > 0:
        if (kernel - stride) % 2 == 0:
            pad_top = pad
            pad_bottom = pad
            pad_left = pad
            pad_right = pad

        else:
            pad_top = pad
            pad_bottom = kernel - stride - pad_top
            pad_left = pad
            pad_right = kernel - stride - pad_left

        x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

    return tf.layers.average_pooling2d(x, pool_size=kernel, strides=stride, padding='VALID')


def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)

    return gap


def z_sample(mean, logvar):
    eps = tf.random_normal(shape=tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)

    return mean + tf.exp(logvar * 0.5) * eps


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)


##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)


def layer_norm(x, scope='layer_norm'):
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm




