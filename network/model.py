from network.ops import *
from keras.models import Model
from network.vgg import vgg_from_t7
import numpy as np
# from tensorflow.keras.models import Model

class ModelNet(object):
    def __init__(self, mode='train', args=None):
        vgg_weights = args['vgg_weights']
        self.batch_size = args['batch_size']
        self.patch_size = args['patch_size']
        self.phase = args['phase']
        self.ablationType = args['ablationType']

        """ data processing """
        self.data_processing(mode, args)

        """ build model """
        self.build_vgg_model(vgg_weights)
        if self.phase == 'test_interpolation':
            self.build_model_interpolation(args)
        else:
            self.build_model(mode, args)

    def build_model_interpolation(self,args):
        n = self.patch_size
        self.patch_mask_c_relu5 = tf.placeholder(tf.int32, [None, None, n * n], name='patch_mask_c_relu5')
        self.patch_mask_s_relu5 = tf.placeholder(tf.int32, [None, None, n * n], name='patch_mask_s_relu5')
        self.patch_mask_c_relu4 = tf.placeholder(tf.int32, [None, None, n * n], name='patch_mask_c_relu4')
        self.patch_mask_s_relu4 = tf.placeholder(tf.int32, [None, None, n * n], name='patch_mask_s_relu4')

        self.patch_mask_segment_c_relu5 = tf.placeholder(tf.int32, [None], name='patch_mask_segment_c_relu5')
        self.patch_mask_segment_c_relu4 = tf.placeholder(tf.int32, [None], name='patch_mask_segment_c_relu4')

        fc_list = self.encoder(self.Ic)

        fs_list0 = self.encoder(self.Is0)
        fcs0 = self.attention_mudule(fc_list, fs_list0)

        fs_list1 = self.encoder(self.Is1)
        fcs1 = self.attention_mudule(fc_list, fs_list1, reuse=True)

        fs_list2 = self.encoder(self.Is2)
        fcs2 = self.attention_mudule(fc_list, fs_list2, reuse=True)

        fs_list3 = self.encoder(self.Is3)
        fcs3 = self.attention_mudule(fc_list, fs_list3, reuse=True)

        a = [[1.0, 0.0, 0.0, 0.0], [0.75, 0.25, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.25, 0.75, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        b = [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.75, 0.25], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.25, 0.75], [0.0, 0.0, 0.0, 1.0]]

        alpha = []
        a = np.array(a)
        b = np.array(b)

        alpha.append(a)
        alpha.append(a * 0.75 + b * 0.25)
        alpha.append(a * 0.5 + b * 0.5)
        alpha.append(a * 0.25 + b * 0.75)
        alpha.append(b)

        self.Ics_interpolation = []
        idx = 0
        for x in alpha:
            for y in x:
                fcs = y[0] * fcs0 + y[1]* fcs1 + y[2] * fcs2 + y[3] * fcs3
                if idx == 0:
                    Ics = self.decoder(fcs, None)
                else:
                    Ics = self.decoder(fcs, None, reuse=True)

                self.Ics_interpolation.append(Ics)
                idx += 1

    def build_model(self, mode, args):

        n = self.patch_size
        self.patch_mask_c_relu5 = tf.placeholder(tf.int32, [None, None, n * n], name='patch_mask_c_relu5')
        self.patch_mask_s_relu5 = tf.placeholder(tf.int32, [None, None, n * n], name='patch_mask_s_relu5')
        self.patch_mask_c_relu4 = tf.placeholder(tf.int32, [None, None, n * n], name='patch_mask_c_relu4')
        self.patch_mask_s_relu4 = tf.placeholder(tf.int32, [None, None, n * n], name='patch_mask_s_relu4')

        self.patch_mask_segment_c_relu5 = tf.placeholder(tf.int32, [None], name='patch_mask_segment_c_relu5')
        self.patch_mask_segment_c_relu4 = tf.placeholder(tf.int32, [None], name='patch_mask_segment_c_relu4')

        fc_list = self.encoder(self.Ic)
        fs_list = self.encoder(self.Is)
        fcs = self.attention_mudule(fc_list, fs_list)
        Ics = self.decoder(fcs, fs_list)
        self.Ics = Ics

        if self.phase == 'test_tradeoff':
            list_alpha = args['alpha_tradeoff']
            self.Ics_tradeoff = []
            fcc = self.attention_mudule(fc_list, fc_list, reuse=True)
            for alpha in list_alpha:
                fcs_cc = alpha * fcs + (1-alpha) * fcc
                self.Ics_tradeoff.append(self.decoder(fcs_cc, fs_list, reuse=True))

        if mode == 'train':
            fcc = self.attention_mudule(fc_list, fc_list, reuse=True)
            Icc = self.decoder(fcc, fc_list, reuse=True)

            fss = self.attention_mudule(fs_list, fs_list, reuse=True)
            Iss = self.decoder(fss, fs_list, reuse=True)

            self.Iss = Iss
            self.Icc = Icc

            fcs_list = self.encoder(Ics)
            fcc_list = self.encoder(Icc)
            fss_list = self.encoder(Iss)

            '''loss ops'''
            w_c = args['weight']['content']
            w_s = args['weight']['style']
            w_cc = args['weight']['cc']
            w_cc_comp = args['weight']['cc_comp']
            w_cc_cons = args['weight']['cc_cons']

            '''perceptual loss'''
            # loss_c = mse(normal(fcs_list[-1]), normal(fc_list[-1])) + mse(normal(fcs_list[-2]), normal(fc_list[-2]))

            # loss_s = calc_style_loss(fcs_list[0], fs_list[0])
            # for i in range (1, 5):
            #     loss_s += calc_style_loss(fcs_list[i],fs_list[i])

            # loss_c = w_c * loss_c
            # loss_s = w_s * loss_s
            # loss_p = loss_c + loss_s

            '''identity loss'''
            l_identity1 = mse(Icc, self.Ic) + mse(Iss, self.Is)

            l_identity2 = mse(fcc_list[0], fc_list[0]) + mse(fss_list[0], fs_list[0])
            for i in range(1, 5):
                l_identity2 += mse(fcc_list[i], fc_list[i]) + mse(fss_list[i], fs_list[i])

            l_identity = (l_identity1 * 50) + (l_identity2 * 1)

            '''cc loss'''
            loss_comp_s, loss_cons_s = get_patch_cc_loss(fcs_list, fs_list)

            loss_cc = (w_cc * w_cc_comp / (w_cc_comp + w_cc_cons)) * loss_comp_s \
                    + (w_cc * w_cc_cons / (w_cc_comp + w_cc_cons)) * loss_cons_s

            loss = l_identity + loss_cc

            self.loss_list = [loss, loss_cc, l_identity]

            '''training ops'''
            self.lr = tf.placeholder(tf.float32, name='learning_rate')
            vars = tf.trainable_variables()
            t_vars = [var for var in vars if 'decoder' in var.name or 'attn' in var.name]

            opt = tf.train.AdamOptimizer(learning_rate=self.lr)
            grads = tf.gradients(loss, t_vars)
            grads, _ = tf.clip_by_global_norm(grads, 1)

            self.optim = opt.apply_gradients(zip(grads, t_vars))
            tf.contrib.training.add_gradients_summaries(zip(grads, t_vars))

            '''summary ops'''
            tf.summary.scalar('loss', loss)
            # tf.summary.scalar('loss_c', loss_c)
            # tf.summary.scalar('loss_s', loss_s)
            tf.summary.scalar('loss_comp_s', loss_comp_s)
            tf.summary.scalar('loss_cons_s', loss_cons_s)
            tf.summary.scalar('l_identity1', l_identity1)
            tf.summary.scalar('l_identity2', l_identity2)

            self.summary = tf.summary.merge_all()

    def attention_mudule(self,fc, fs, scope='attn_module', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):

            fc_5_cons = self.attention_block(fc[-1], fs[-1], self.patch_mask_c_relu5, self.patch_mask_s_relu5, self.patch_mask_segment_c_relu5, type='cons', scope='cons_5', reuse=reuse)
            fc_5_comp = self.attention_block(fc[-1], fs[-1], self.patch_mask_c_relu5, self.patch_mask_s_relu5, self.patch_mask_segment_c_relu5, type='comp', scope='comp_5', reuse=reuse)
            fc_5 = self.joint_analysis(fc_5_cons, fc_5_comp, fc[-1], scope='joint_5')

            fc_4_cons = self.attention_block(fc[-2], fs[-2], self.patch_mask_c_relu4, self.patch_mask_s_relu4, self.patch_mask_segment_c_relu4, type='cons', scope='cons_4',reuse=reuse)
            fc_4_comp = self.attention_block(fc[-2], fs[-2], self.patch_mask_c_relu4, self.patch_mask_s_relu4, self.patch_mask_segment_c_relu4, type='comp', scope='comp_4',reuse=reuse)
            fc_4 = self.joint_analysis(fc_4_cons, fc_4_comp, fc[-2], scope='joint_4')

            fc_4_plus_5 = fc_4 + up_sample(fc_5)
            fm = conv(fc_4_plus_5, 512, kernel=3, stride=1, pad=1, pad_type='reflect', scope='out_conv')

            return fm


    def joint_analysis(self, f_cons, f_comp, fc, scope='attn_joint', eps=1e-5, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            C = 512

            fo_comp = self.joint_blk(f_cons, f_comp, scope='joint_blk')
            fo_cons = self.joint_blk(f_comp, f_cons, scope='joint_blk', filter=True, reuse=True)

            fo_comp = relu(fo_comp)
            fo_cons = relu(fo_cons)

            fo = tf.sqrt(tf.maximum(tf.multiply(fo_comp, fo_cons), eps))
            fo = conv(fo, C, kernel=1, stride=1, pad=0, pad_type='reflect', scope='mul_conv')
            fo = fo + fc

            return fo

    def joint_blk(self, f1, f2, scope='attn_joint_blk', filter=False, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            C = 512
            f = conv(f1, C, kernel=1, stride=1, pad=0, pad_type='reflect', scope='f_conv')  # [b, hc, wc, c']
            g = conv(f2, C, kernel=1, stride=1, pad=0, pad_type='reflect', scope='g_conv')  # [b, hs, ws, c']
            h = conv(f2, C, kernel=1, stride=1, pad=0, pad_type='reflect', scope='h_conv')  # [b, hs, ws, c']

            # TODO check the computation of s -- the order of f and g?
            s = tf.matmul(hw_flatten(f), hw_flatten(g), transpose_b=True)  # [b, hcwc, hsws]

            attn = tf.nn.softmax(s, dim=-1)

            o = tf.matmul(attn, hw_flatten(h))  # [b, hcwc, hsws] * [b, hsws, c]--> [b, hcwc, c]
            o = tf.reshape(o, shape=tf.shape(f2))  # [bs, h, w, c]

            if filter:
                f1_norm = tf.norm(f1, axis=-1, keep_dims=True)
                filter_mask = tf.greater_equal(f1_norm, 0.01)
                filter_mask = tf.tile(filter_mask, multiples=(1, 1, 1, C))
                filter_mask = tf.cast(filter_mask, tf.float32)
                o = tf.multiply(filter_mask, o)

            o = conv(o + f2, C, kernel=1, stride=1, pad=0, pad_type='reflect', scope='out_conv')

            return o

    def attention_block(self, fc, fs, patch_mask_c, patch_mask_s, patch_mask_segment_c, type='cons', eps=1e-5, scope='attn_blk', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            N = self.patch_size
            C = 512
            B = self.batch_size
            CN = C*N
            pad = N - 1
            _, H, W, _ = tf.unstack(tf.shape(fc))

            fc_n = normal(fc)
            fs_n = normal(fs)

            f = conv(fc_n, C, kernel=1, stride=1, pad=0, pad_type='reflect', scope='f_conv')  # [b, hc, wc, c']
            g = conv(fs_n, C, kernel=1, stride=1, pad=0, pad_type='reflect', scope='g_conv')  # [b, hs, ws, c']
            h = conv(fs, C, kernel=1, stride=1, pad=0, pad_type='reflect', scope='h_conv')  # [b, hs, ws, c']

            # padding
            f_pad = pad_reflect(f, padding=pad, mode='SYMMETRIC')
            g_pad = pad_reflect(g, padding=pad, mode='SYMMETRIC')
            h_pad = pad_reflect(h, padding=pad, mode='SYMMETRIC')

            f_patch_0 = tf.reshape(f_pad, shape=(B, -1, C))
            g_patch_0 = tf.reshape(g_pad, shape=(B, -1, C))
            h_patch_0 = tf.reshape(h_pad, shape=(B, -1, C))

            # Computing patches
            f_patch = tf.gather(f_patch_0, indices=patch_mask_c, axis=1)
            g_patch = tf.gather(g_patch_0, indices=patch_mask_s, axis=1)
            h_patch = tf.gather(h_patch_0, indices=patch_mask_s, axis=1)

            f_patch = tf.reshape(f_patch, shape=(B, -1, CN))
            g_patch = tf.reshape(g_patch, shape=(B, -1, CN))
            h_patch = tf.reshape(h_patch, shape=(B, -1, CN))


            ws = tf.matmul(f_patch, g_patch, transpose_b=True)  # [b, hcwc, hsws]

            if type == 'comp':
                ws = tf.nn.softmax(ws, axis=-2)
                o_patch = tf.matmul(ws, h_patch)
                # # Clip the weight matrix outputted by softmax
                # p_ws = tf.reduce_sum(ws, axis=-1)
                # p_ws = tf.clip_by_value(p_ws, clip_value_min=1.0, clip_value_max=10000.0)
                # p_ws = tf.divide(1.0, p_ws)
                # p_ws = tf.reshape(p_ws, shape=(B, -1, 1))
                # p_ws = tf.tile(p_ws, multiples=[1, 1, CN])
                # o_patch = tf.multiply(o_patch, p_ws)

            elif type == 'cons':
                ws = tf.nn.softmax(ws, axis=-1)
                o_patch = tf.matmul(ws, h_patch) # [b, hcwc, hsws] * [b, hsws, cn]--> [b, hcwc, cn]

            # Merge the patch information into pixels
            o_patch = tf.reshape(o_patch, shape=(-1, C))
            o = tf.unsorted_segment_sum(o_patch, patch_mask_segment_c, B * (H + 2 * pad) * (W + 2 * pad))

            o = tf.reshape(o, shape=tf.shape(f_pad))  # [bs, h, w, c]
            o = tf.slice(o, [0, pad, pad, 0], [B, H, W, C])
            o = conv(o, C, kernel=1, stride=1, pad=0, pad_type='reflect', scope='out_conv')
            return o

    def decoder(self, x, fs, scope='decoder', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x, 256, kernel=3, stride=1, pad=1, pad_type='reflect', use_relu=True, scope='conv_4_0')
            x = up_sample(x)

            x = conv(x, 256, kernel=3, stride=1, pad=1, pad_type='reflect', use_relu=True, scope='conv_3_3')

            # x = self.fuse(x, fs[2], channel=256, scope='fuse_3')
            # x = adain_norm(x, fs[2]) + x

            x = conv(x, 256, kernel=3, stride=1, pad=1, pad_type='reflect', use_relu=True, scope='conv_3_2')
            x = conv(x, 256, kernel=3, stride=1, pad=1, pad_type='reflect', use_relu=True, scope='conv_3_1')
            x = conv(x, 128, kernel=3, stride=1, pad=1, pad_type='reflect', use_relu=True, scope='conv_3_0')
            x = up_sample(x)

            x = conv(x, 128, kernel=3, stride=1, pad=1, pad_type='reflect', use_relu=True, scope='conv_2_1')

            # x = self.fuse(x, fs[1], channel=128, scope='fuse_2')
            # x = adain_norm(x, fs[1]) + x

            x = conv(x, 64, kernel=3, stride=1, pad=1, pad_type='reflect', use_relu=True, scope='conv_2_0')
            x = up_sample(x)

            x = conv(x, 64, kernel=3, stride=1, pad=1, pad_type='reflect', use_relu=True, scope='conv_1_1')
            x = conv(x, 3, kernel=3, stride=1, pad=1, pad_type='reflect', use_relu=False, scope='conv_1_0')
            return x

    def fuse(self, x, fs, channel=512, scope='fuse', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            _, w, h, _ = tf.unstack(tf.shape(x))

            weight = tf.get_variable(name='w', shape=[channel, channel], initializer=weight_init_variable)
            weight = tf.tile(tf.expand_dims(weight, axis=0), [self.batch_size, 1, 1])

            x1 = tf.reshape(x, shape=[self.batch_size, -1, channel])
            gram_fs = tf.sqrt(gram_matrix(fs))
            code_fusion = tf.matmul(weight, gram_fs)
            code_fusion = tf.matmul(x1, code_fusion)
            code_fusion = tf.reshape(code_fusion, shape=[self.batch_size, w, h, channel])
            code_fusion = tf.concat([code_fusion, x], axis=-1)

            return code_fusion

    # def build_vgg_model(self, vgg_weights):
    #     with tf.variable_scope('vgg_model'):
    #         self.vgg_model = vgg_from_t7(vgg_weights, target_layer='relu5_1')
    #
    #         # Build style model for blockX_conv1 tensors for X:[1,2,3,4]
    #         relu_layers = [ 'relu1_1',
    #                         'relu2_1',
    #                         'relu3_1',
    #                         'relu4_1',
    #                         'relu5_1']
    #
    #         style_layers = [self.vgg_model.get_layer(l).output for l in relu_layers]
    #         self.encoder = Model(inputs=self.vgg_model.input, outputs=style_layers)

    def build_vgg_model(self, vgg_weights):
        vgg_model = vgg_from_t7(vgg_weights, target_layer='relu5_1')
        relu_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        # model_layers = [vgg_model.get_layer(l).output for l in relu_layers]
        # self.encoder = Model(inputs=vgg_model.input, outputs=model_layers)

        style_layers = [vgg_model.get_layer(l).output for l in relu_layers]
        self.encoder = Model(inputs=vgg_model.input, outputs=style_layers)

    def data_processing(self, mode, args):
        dir_style = args['data']['dir_style']
        dir_content = args['data']['dir_content']

        if mode == 'train':
            img_size = args['img_size']
            self.Ic, self.Is, self.file_c, self.file_s = processing_data(dir_style, dir_content, self.batch_size, img_size[0],
                                                       img_size[1], img_size[2])

        else:
            if self.phase == 'test_interpolation':
                self.batch_size = 1
                self.Ic = tf.placeholder(tf.float32, [1, None, None, 3], name='test_image')
                self.Is0 = tf.placeholder(tf.float32, [1, None, None, 3], name='test_style0')
                self.Is1 = tf.placeholder(tf.float32, [1, None, None, 3], name='test_style1')
                self.Is2 = tf.placeholder(tf.float32, [1, None, None, 3], name='test_style2')
                self.Is3 = tf.placeholder(tf.float32, [1, None, None, 3], name='test_style3')
            else:
                self.batch_size = 1
                self.Ic = tf.placeholder(tf.float32, [1, None, None, 3], name='test_image')
                self.Is = tf.placeholder(tf.float32, [1, None, None, 3], name='test_style')

    def get_style_loss(self, fake_maps, style_maps):
        gram_losses = []
        for s_map, d_map in zip(style_maps, fake_maps):
            s_gram = gram_matrix(s_map)
            d_gram = gram_matrix(d_map)
            gram_loss = mse(d_gram, s_gram)
            gram_losses.append(gram_loss)
        style_loss = tf.reduce_sum(gram_losses)
        return style_loss