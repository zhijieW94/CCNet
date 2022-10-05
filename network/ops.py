from glob import glob
import tensorflow as tf
import tensorflow.contrib as tf_contrib
from tensorflow.contrib.data import batch_and_drop_remainder
from data_processing.data_processing import ImageData

weight_init = tf_contrib.layers.variance_scaling_initializer(uniform=True)
weight_init_conv = tf_contrib.layers.variance_scaling_initializer(factor=1/27 ,mode='FAN_IN',uniform=True)
# weight_init_conv = tf.truncated_normal_initializer(stddev=0.001, seed=250)
# weight_init_conv = tf.truncated_normal_initializer(stddev=0.01, seed=250)
weight_init_variable = tf.random_uniform_initializer(minval=0.0, maxval=0.001, seed = 20)

##################################################################################
# Data processing
##################################################################################
def processing_data(dir_B, dir_A, batch_size, h, w, ch):
    trainA_dataset = glob(dir_A + '/*.*')
    trainB_dataset = glob(dir_B + '/*.*')

    dataset_num = max(len(trainA_dataset), len(trainB_dataset))

    Image_Data_Class = ImageData(h, w, ch)

    trainA = tf.data.Dataset.from_tensor_slices(trainA_dataset)
    trainB = tf.data.Dataset.from_tensor_slices(trainB_dataset)

    trainA = trainA.prefetch(batch_size).shuffle(dataset_num).map(Image_Data_Class.image_processing, num_parallel_calls=8).apply(batch_and_drop_remainder(batch_size)).repeat()
    trainB = trainB.prefetch(batch_size).shuffle(dataset_num).map(Image_Data_Class.image_processing, num_parallel_calls=8).apply( batch_and_drop_remainder(batch_size)).repeat()

    trainA_iterator = trainA.make_one_shot_iterator()
    trainB_iterator = trainB.make_one_shot_iterator()

    domain_A_info = trainA_iterator.get_next()
    domain_B_info = trainB_iterator.get_next()

    img_A, file_A = domain_A_info
    img_B, file_B = domain_B_info

    return img_A, img_B, file_A, file_B

def processing_data_test(img, layers_num):
    img_list = []
    b, h, w, c = tf.unstack(tf.shape(img))
    for layer in range(layers_num):
        if layer == layers_num-1:
            img_list.append(img)
        else:
            factor = (layers_num - 1 - layer) * 2
            resize_shape = (h // factor, w // factor)
            img_temp = tf.image.resize_images(img, resize_shape)
            img_list.append(img_temp)
    return img_list

def processing_By_config(img_A, img_B, h, w, layers_num, layer):
    factor = (layers_num-1-layer)*2
    resize_shape = (tf.constant(h // factor), tf.constant(w // factor))
    img_A = tf.image.resize_images(img_A, resize_shape)
    img_B = tf.image.resize_images(img_B, resize_shape)
    return img_A, img_B

def data_processing(args, mode):
    if mode == 'test':
        Ic = tf.placeholder(tf.float32, [1, None, None, 3], name='Ic')
        Is = tf.placeholder(tf.float32, [1, None, None, 3], name='Is')

        file_c = None
        file_s = None

    else:
        batch_size = args['batch_size']
        dir_style = args['data']['dir_style']
        dir_content = args['data']['dir_content']

        img_size = args['img_size']
        Ic, Is, file_c, file_s = processing_data(dir_style, dir_content, batch_size, img_size[0], img_size[1], img_size[2])

    return Ic, Is, file_c, file_s

# def data_processing(batch_size, nums_gpu, args):
#     dir_style = args['data']['dir_style']
#     dir_content = args['data']['dir_content']
#
#     img_size = args['img_size']
#     Ic, Is, file_c, file_s = processing_data(dir_style, dir_content, batch_size, img_size[0],
#                                              img_size[1], img_size[2])
#
#     batch_size = batch_size // nums_gpu
#     Ic = tf.reshape(Ic, shape=(nums_gpu, batch_size, 256, 256, 3))
#     Is = tf.reshape(Is, shape=(nums_gpu, batch_size, 256, 256, 3))
#     return Ic, Is, file_c, file_s

##################################################################################
# Layer
##################################################################################
def pad_reflect(x, padding=1, mode='REFLECT'):
    return tf.pad(
      x, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
      mode=mode)

def conv(x, channels, kernel=4, stride=2, pad=0, dilation_rate=(1, 1), pad_type='zero',
         use_bias=True, scope='conv', use_relu=False, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        padding = 'valid'
        if pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
        if pad_type == 'same':
            padding = 'same'
        # x = tf.layers.conv2d(inputs=x, filters=channels,dilation_rate=dilation_rate, padding=padding,
        #                      kernel_size=kernel, kernel_initializer=weight_init_conv,
        #                      strides=stride, use_bias=use_bias, bias_initializer=weight_init_conv)
        x = tf.layers.conv2d(inputs=x, filters=channels, dilation_rate=dilation_rate, padding=padding,
                             kernel_size=kernel, strides=stride, use_bias=use_bias)
        if use_relu:
            x = relu(x)
        return x

def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def flatten(x) :
    return tf.layers.flatten(x)

def linear(x, units, use_bias=True, scope='linear'):
    with tf.variable_scope(scope):
        # x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, use_bias=use_bias)
        return x
##################################################################################
# Normalization function
##################################################################################
def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x, epsilon=1e-05, center=True, scale=True, scope=scope)

# def normal(x, scope='normal'):
#     return tf_contrib.layers.instance_norm(x, epsilon=1e-05, center=False, scale=False, scope=scope)

def layer_norm(x, scope='layer_norm') :
    return tf_contrib.layers.layer_norm(x, center=True, scale=True, scope=scope)

#AdaIN
def adain_norm(c, s):
    epsilon = 1e-10
    mean_c, std_c = calc_mean_std(c, epsilon)
    mean_s, std_s = calc_mean_std(s, epsilon)
    fcs = std_s * (c - mean_c) / std_c + mean_s
    return fcs

##################################################################################
# Activation function
##################################################################################
def lrelu(x, alpha=0.1):
    return tf.nn.leaky_relu(x, alpha)

def relu(x, scope=None):
    return tf.nn.relu(x, name=scope)

def tanh(x):
    return tf.tanh(x)

##################################################################################
# Residual-block
##################################################################################
def res_block_simple(x_input, use_bias=True, scope='simple_resblock', use_norm_layer=False, normType='Layer', reuse=False) :
    with tf.variable_scope(scope, reuse=reuse):
        channel = x_input.get_shape()[-1]
        x_init = x_input
        with tf.variable_scope('res1'):
            x = conv(x_init, channel, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            if use_norm_layer:
                if normType == 'Layer':
                    x = layer_norm(x)
                elif normType == 'IN':
                    x = instance_norm(x)

            x = lrelu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channel, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            if use_norm_layer:
                if normType == 'Layer':
                    x = layer_norm(x)
                elif normType == 'IN':
                    x = instance_norm(x)

        return x + x_init

def res_block(x_input, channels=-1, reuse=False, use_norm_layer=True, normType='Layer', use_relu=True, bottle_neck=False, bottle_down_sample=True, scope="res_block"):
    with tf.variable_scope(scope, reuse=reuse):
        c_i = x_input.get_shape()[-1]

        if channels == -1:
            channels = c_i

        if not bottle_neck:
            first_output_channel = channels / 4
            second_output_channel = channels / 4
            third_output_channel = channels

            first_conv_output = conv(x_input, channels=first_output_channel, kernel=1, stride=1, pad_type='reflect',
                                     use_bias=False, scope='res_conv_0')
            if use_norm_layer:
                if normType == 'Layer':
                    first_conv_output = layer_norm(first_conv_output, scope='res_conv_norm_layer_0')
                elif normType == 'IN':
                    first_conv_output = instance_norm(first_conv_output, scope='res_conv_norm_layer_0')

            first_conv_output = lrelu(first_conv_output)

            second_conv_output = conv(first_conv_output, channels=second_output_channel, kernel=3, stride=1, pad_type='reflect', pad=1,
                                      use_bias=False, scope='res_conv_1')
            if use_norm_layer:
                if normType == 'Layer':
                    second_conv_output = layer_norm(second_conv_output, scope='res_conv_norm_layer_1')
                elif normType == 'IN':
                    second_conv_output = instance_norm(second_conv_output, scope='res_conv_norm_layer_1')


            second_conv_output = lrelu(second_conv_output)

            third_conv_output = conv(second_conv_output, channels=third_output_channel, kernel=1, stride=1, pad_type='reflect',
                                     use_bias=False, scope='res_conv_2')
            if use_norm_layer:
                if normType == 'Layer':
                    third_conv_output = layer_norm(third_conv_output, scope='res_conv_norm_layer_2')
                elif normType == 'IN':
                    third_conv_output = instance_norm(third_conv_output, scope='res_conv_norm_layer_2')

            if channels != c_i:
                x_input = conv(x_input, channels=third_output_channel, kernel=1, stride=1, pad_type='reflect', use_bias=False,scope='res_conv3')

            res_output = x_input + third_conv_output
            if use_relu:
                 res_output = lrelu(res_output)
        else:
            first_output_channel = channels / 2
            second_output_channel = channels / 2
            third_output_channel = channels

            first_conv_output = conv(x_input, channels=first_output_channel, kernel=1, stride=2, pad_type='reflect',
                                     use_bias=False, scope='res_conv_0')
            if use_norm_layer:
                if normType == 'Layer':
                    first_conv_output = layer_norm(first_conv_output, scope='res_conv_norm_layer_0')
                elif normType == 'IN':
                    first_conv_output = instance_norm(first_conv_output, scope='res_conv_norm_layer_0')

            first_conv_output = lrelu(first_conv_output)

            second_conv_output = conv(first_conv_output, channels=second_output_channel, kernel=3, stride=1,
                                      pad_type='reflect', pad=1, use_bias=False, scope='res_conv_1')
            if use_norm_layer:
                if normType == 'Layer':
                    second_conv_output = layer_norm(second_conv_output, scope='res_conv_norm_layer_1')
                elif normType == 'IN':
                    second_conv_output = instance_norm(second_conv_output, scope='res_conv_norm_layer_1')

            second_conv_output = lrelu(second_conv_output)

            third_conv_output = conv(second_conv_output, channels=third_output_channel, kernel=1, stride=1,
                                     pad_type='reflect', use_bias=False, scope='res_conv_2')
            if use_norm_layer:
                if normType == 'Layer':
                    third_conv_output = layer_norm(third_conv_output, scope='res_conv_norm_layer_2')
                elif normType == 'IN':
                    third_conv_output = instance_norm(third_conv_output, scope='res_conv_norm_layer_2')

            if bottle_down_sample:
                left_output = conv(x_input, channels=channels, kernel=1, stride=2, pad_type='zero', use_bias=False, scope='res_left_conv_0')
            else:
                left_output = conv(x_input, channels=channels, kernel=1, stride=1, pad_type='zero', use_bias=False, scope='res_left_conv_0')

            if use_norm_layer:
                if normType == 'Layer':
                    left_output = layer_norm(left_output, scope='res_conv_norm_layer_left')
                elif normType == 'IN':
                    left_output = instance_norm(left_output, scope='res_conv_norm_layer_left')


            res_output = left_output + third_conv_output
            if use_relu:
                res_output = lrelu(res_output)

        return res_output

##################################################################################
# Sampling
##################################################################################
def up_sample(x, scale_factor=2):
    _, h, w, _ = tf.unstack(tf.shape(x))
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def down_sample(x, scale_factor=2):
    _, h, w, _ = tf.unstack(tf.shape(x))
    new_size = [h // scale_factor, w // scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

##################################################################################
# Loss
##################################################################################
def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss
def L2_loss(x):
    return tf.reduce_mean(tf.square(x))

def mse(x,y):
    '''Mean Squared Error'''
    return tf.reduce_mean(tf.square(x - y))

def sse(x,y):
    '''Sum of Squared Error'''
    return tf.reduce_sum(tf.square(x - y))

def gram_matrix(feature_maps):
    """Computes the Gram matrix for a set of feature maps."""
    batch_size, height, width, channels = tf.unstack(tf.shape(feature_maps))
    denominator = tf.to_float(height * width)
    feature_maps = tf.reshape(feature_maps, tf.stack([batch_size, height * width, channels]))
    matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
    return matrix / denominator

def gram_matrixs(vgg_maps):
    enc_style = [gram_matrix(s_map) for s_map in vgg_maps]
    return enc_style

def tf_cov(x,reduceMean = True):
    batch_size, height, width, channels = tf.unstack(tf.shape(x))
    x = tf.reshape(x, tf.stack([batch_size, height * width, channels]))
    mc = tf.reduce_mean(x, axis=1, keepdims=True)
    if reduceMean:
        fc = x - mc
    else:
        fc = x
    fcfc = tf.matmul(tf.transpose(fc,[0,2,1]),fc) / (tf.cast(tf.shape(fc)[1], tf.float32))
    return fcfc

def cov_matrixs(vgg_maps):
    enc_style = [tf_cov(s_map) for s_map in vgg_maps]
    return enc_style

def calc_style_loss(input, target):
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return mse(input_mean, target_mean) + \
           mse(input_std, target_std)

def calc_mean_std(x, eps=1e-10):
    # eps is a small value added to the variance to avoid divide-by-zero.
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return mean, tf.sqrt(tf.maximum(var,eps))


def normal(feat, eps=1e-10):
    feat_mean, feat_std= calc_mean_std(feat, eps)
    normalized=(feat-feat_mean)/feat_std
    return normalized


def get_content_loss(fcs_list, fc_list):
    content_loss = 0.5 * (mse(fcs_list[-1], fc_list[-1]) + mse(fcs_list[-2], fc_list[-2]))
    return content_loss


def get_style_loss_mu_sigma(fake_maps, style_maps):
    style_losses = []

    for s_map, d_map in zip(style_maps, fake_maps):
        style_losses.append(calc_style_loss(d_map, s_map))

    style_loss = tf.reduce_mean(style_losses)

    return style_loss


def get_style_loss_gramMatrix(fake_maps, style_maps):
    gram_losses = []

    for s_map, d_map in zip(style_maps, fake_maps):
        s_gram = gram_matrix(s_map)
        d_gram = gram_matrix(d_map)
        gram_loss = mse(d_gram, s_gram)
        gram_losses.append(gram_loss)

    style_loss = tf.reduce_mean(gram_losses)

    return style_loss


def get_attn_loss(fcs_b1, fcs_b2, fc, fs, weight):
    b, h, w, c = tf.unstack(tf.shape(fc))

    fcs_b1 = tf.reshape(fcs_b1, shape=(b, -1, c))
    fcs_b2 = tf.reshape(fcs_b2, shape=(b, -1, c))
    fc = tf.reshape(fc, shape=(b, -1, c))
    fs = tf.reshape(fs, shape=(b, -1, c))

    fcs_b1 = tf.divide(fcs_b1, tf.norm(fcs_b1, axis=2, keep_dims=True))
    fcs_b2 = tf.divide(fcs_b2, tf.norm(fcs_b2, axis=2, keep_dims=True))
    fc = tf.divide(fc, tf.norm(fc, axis=2, keep_dims=True))
    fs = tf.divide(fs, tf.norm(fs, axis=2, keep_dims=True))

    #loss_completeness
    d_c_b1 = 1 - tf.matmul(fcs_b1, fc, transpose_b=True)
    d_s_b1 = 1 - tf.matmul(fcs_b1, fs, transpose_b=True)

    d_c_b1_min = tf.reduce_mean(tf.reduce_min(d_c_b1, axis=1))
    d_s_b1_min = tf.reduce_mean(tf.reduce_min(d_s_b1, axis=1))

    #loss_consistency
    d_c_b2 = 1 - tf.matmul(fcs_b2, fc, transpose_b=True)
    d_s_b2 = 1 - tf.matmul(fcs_b2, fs, transpose_b=True)

    d_c_b2_min = tf.reduce_mean(tf.reduce_min(d_c_b2, axis=2))
    d_s_b2_min = tf.reduce_mean(tf.reduce_min(d_s_b2, axis=2))

    return d_c_b1_min + d_s_b1_min * weight, d_c_b2_min + d_s_b2_min * weight

def get_patch_cc_loss(fcs_list, fs_list):
    loss_cc = [[0.0, 0.0]]

    # patch size = 1
    loss_cc.append(get_cc_loss(hw_flatten(fcs_list[-1]), hw_flatten(fs_list[-1])))
    loss_cc.append(get_cc_loss(hw_flatten(fcs_list[-2]), hw_flatten(fs_list[-2])))
    loss_cc.append(get_cc_loss(hw_flatten(fcs_list[-3]), hw_flatten(fs_list[-3])))




    # Relu_5 & patch size = 3
    patch3_fcs_5 = tf.extract_image_patches(images=fcs_list[-1], ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
    patch3_fs_5 = tf.extract_image_patches(images=fs_list[-1], ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')

    B, _, _, C = tf.unstack(tf.shape(fcs_list[-1]))
    patch3_fcs_5 = tf.reshape(patch3_fcs_5, shape=(B, -1, C * 9))
    patch3_fs_5 = tf.reshape(patch3_fs_5, shape=(B, -1, C * 9))

    loss_cc.append(get_cc_loss(patch3_fcs_5, patch3_fs_5))

    # Relu_4 & patch size = 3
    patch3_fcs_4 = tf.extract_image_patches(images=fcs_list[-2], ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
    patch3_fs_4 = tf.extract_image_patches(images=fs_list[-2], ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')

    B, _, _, C = tf.unstack(tf.shape(fcs_list[-2]))
    patch3_fcs_4 = tf.reshape(patch3_fcs_4, shape=(B, -1, C * 9))
    patch3_fs_4 = tf.reshape(patch3_fs_4, shape=(B, -1, C * 9))

    loss_cc.append(get_cc_loss(patch3_fcs_4, patch3_fs_4))

    # Relu_3 & patch size = 3
    patch3_fcs_3 = tf.extract_image_patches(images=fcs_list[-3], ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
    patch3_fs_3 = tf.extract_image_patches(images=fs_list[-3], ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')

    B, _, _, C = tf.unstack(tf.shape(fcs_list[-3]))
    patch3_fcs_3 = tf.reshape(patch3_fcs_3, shape=(B, -1, C * 9))
    patch3_fs_3 = tf.reshape(patch3_fs_3, shape=(B, -1, C * 9))

    loss_cc.append(get_cc_loss(patch3_fcs_3, patch3_fs_3))




    loss_cc = tf.reduce_mean(loss_cc, axis=0)
    loss_comp_s, loss_cons_s = tf.split(loss_cc, 2)

    loss_comp_s = tf.reshape(loss_comp_s, [])
    loss_cons_s = tf.reshape(loss_cons_s, [])

    return loss_comp_s, loss_cons_s

def get_cc_loss(fcs, fs):

    fcs = tf.divide(fcs, tf.norm(fcs, axis=2, keep_dims=True))
    fs = tf.divide(fs, tf.norm(fs, axis=2, keep_dims=True))

    d_s = 1 - tf.matmul(fcs, fs, transpose_b=True)

    loss_comp_s = tf.reduce_mean(tf.reduce_min(d_s, axis=1))
    loss_cons_s = tf.reduce_mean(tf.reduce_min(d_s, axis=2))

    return loss_comp_s, loss_cons_s


def distance_matrix(array1, array2):
    """
    arguments:
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
            , it's size: (num_point, num_point)
    """
    b, num_point1, num_features = array1.shape
    b, num_point2, num_features = array2.shape
    expanded_array1 = tf.tile(array1, (1, num_point2, 1))
    expanded_array2 = tf.reshape(
            tf.tile(tf.expand_dims(array2, 2),
                    (1, 1, num_point1, 1)),
            (b, -1, num_features))
    distances = tf.norm(expanded_array1-expanded_array2, axis=2)
    distances = tf.reshape(distances, (b, num_point2, num_point1))
    return distances

def av_dist(array1, array2):
    """
    arguments:
        array1, array2: both size: (num_points, num_feature)
    returns:
        distances: size: (1,)
    """
    distances = distance_matrix(array1, array2)
    distances = tf.reduce_min(distances, axis=2)
    distances = tf.reduce_mean(distances)
    return distances

#every point in array2 to array1
def chamfer_distance_tf(array1, array2):
    [b, _, _, c] = array1.get_shape().as_list()
    array1 = tf.reshape(array1, shape=(b, -1, c))
    array2 = tf.reshape(array2, shape=(b, -1, c))
    dist = av_dist(array1, array2)
    return dist

##################################################################################
# Gradient
##################################################################################
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads



