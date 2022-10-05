from network.model import ModelNet as model_net
from utils import *
from network.ops import *
from data_processing.data_processing import save_images
import numpy as np
# from tensorflow.python import debug as tf_debug

class Train(object):
    def __init__(self, args):
        print('================================================================')
        print('================         SANet         =========================')
        print('================================================================')
        self.gpu_id = args['GPU_ID']
        self.epoch = args['epoch']
        self.iteration = args['iteration']
        self.batch_size = args['batch_size']

        self.init_lr   = args['lr']
        self.lr_decay  = args['lr_decay']

        self.print_freq = args['freq_print']
        self.save_freq  = args['freq_save']
        self.log_freq  = args['freq_log']

        self.img_size = args['img_size']
        self.patch_size = args['patch_size']

        self.checkpoint_dir_load = args['dir_checkpoint']

        '''build model'''
        self.model = model_net(mode='train', args=args)

        ''' build folders for saving results'''
        self.log_dir, self.config_dir, self.sample_dir, self.checkpoint_dir, _ = mkdir_output_train(args)

        ''' load model'''
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = False))
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        self.writer = tf.summary.FileWriter(self.log_dir + '/', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        could_load, checkpoint_counter = self.loadCheckpoint()

        if could_load:
            # self.start_epoch = (int)(checkpoint_counter / self.iteration)
            # self.start_batch_id = checkpoint_counter - self.start_epoch * self.iteration
            # self.counter = checkpoint_counter

            self.start_epoch = 0
            self.start_batch_id = 0
            self.counter = 0
            print(" [*] Load SUCCESS")
        else:
            self.start_epoch = 0
            self.start_batch_id = 0
            self.counter = 0
            print(" [!] Load failed...")

        vars = tf.trainable_variables()
        t_vars = [var for var in vars if 'decoder' in var.name or 'attn' in var.name]
        self.saver = tf.train.Saver(var_list=t_vars)

        self.fetches = {
            'Ic': self.model.Ic,
            'Is': self.model.Is,
            'Ics': self.model.Ics,
            'Icc': self.model.Icc,
            'Iss': self.model.Iss,
            'loss_list': self.model.loss_list,
            'train': self.model.optim,
            'summary': self.model.summary
        }

        self.patch_mask_relu5, self.patch_mask_segment_relu5, self.patch_mask_relu4, self.patch_mask_segment_relu4 = self.get_patch_mask()

    def train(self):
        for epoch in range(self.start_epoch, self.epoch):
            for idx in range(self.start_batch_id, self.iteration):
                lr = self.init_lr / (1 + self.counter * self.lr_decay)
                start_time = time.time()
                results = self.sess.run(self.fetches, feed_dict={self.model.lr: lr,

                                                                 self.model.patch_mask_c_relu5: self.patch_mask_relu5,
                                                                 self.model.patch_mask_s_relu5: self.patch_mask_relu5,
                                                                 self.model.patch_mask_c_relu4: self.patch_mask_relu4,
                                                                 self.model.patch_mask_s_relu4: self.patch_mask_relu4,

                                                                 self.model.patch_mask_segment_c_relu5: self.patch_mask_segment_relu5,
                                                                 self.model.patch_mask_segment_c_relu4: self.patch_mask_segment_relu4

                                                                 })

                self.counter += 1

                #print losses
                print(
                    "GPU_id:[%s] Epoch: [%2d] [%6d/%6d] time: %4.4f loss: %.8f l_c: %.8f l_s: %.8f  " \
                    % (self.gpu_id, epoch, idx, self.iteration, time.time() - start_time,
                       results['loss_list'][0], results['loss_list'][1], results['loss_list'][2]))

                #save summary
                if np.mod(idx, self.log_freq) == 0:
                    self.writer.add_summary(results['summary'], self.counter)

                # save images
                if np.mod(idx+1, self.print_freq) == 0:
                    list_img_temp = []
                    for img_A, img_B, img_C, img_D, img_E in zip(results['Ic'], results['Icc'], results['Ics'], results['Iss'], results['Is']):
                        list_img_temp.append(img_A)
                        list_img_temp.append(img_B)
                        list_img_temp.append(img_C)
                        list_img_temp.append(img_D)
                        list_img_temp.append(img_E)

                    array_img_out = np.array(list_img_temp, dtype=np.float32)
                    num = 5
                    save_images(array_img_out, [self.batch_size * num, int(num)],
                                '{}/{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1))

                #save model
                if np.mod(idx, self.save_freq) == 0:
                    self.saveCheckpoint(self.counter)

            self.start_batch_id = 0

            # save model for final step
            self.saveCheckpoint(self.counter)
        print("finish...!")

    def loadCheckpoint(self):
        import re
        print(" [*] Reading checkpoints...")
        try:
            counter = 0
            dir_checkpoint = self.checkpoint_dir_load

            ckpt = tf.train.get_checkpoint_state(dir_checkpoint)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                vars = tf.trainable_variables()

                t_vars = [var for var in vars if 'decoder' in var.name or 'attn' in var.name]

                self.saver = tf.train.Saver(var_list=t_vars)
                self.saver.restore(self.sess, os.path.join(dir_checkpoint, ckpt_name))

                counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
                print(" [*] Success to read {}".format(ckpt_name))

            print(" [*] Success to load checkpoint!!!")
            return True, counter
        except:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def saveCheckpoint(self, step):
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'SANet.model'), global_step=step)


    def get_patch_mask(self):
        n = self.patch_size
        pad = n - 1

        # relu_5
        h = int(self.img_size[0] / np.power(2, 4) + 2*pad)
        w = int(self.img_size[1] / np.power(2, 4) + 2*pad)
        patch_mask_relu5 = np.zeros((h - n + 1, w - n + 1, n*n), dtype=int)

        for i in range(h - n + 1):
            for j in range(w - n + 1):
                for a in range(n):
                    for b in range(n):
                        k = a * n + b
                        patch_mask_relu5[i, j, k] = (i + a) * w + j + b

        patch_mask_temp = np.reshape(patch_mask_relu5, newshape=[-1])
        patch_mask_segment_relu5 = []
        for i in range(self.batch_size):
            patch_mask_segment_relu5.append(patch_mask_temp + i * h * w)
        patch_mask_segment_relu5 = np.reshape(patch_mask_segment_relu5, newshape=[-1])

        # relu_4
        h = int(self.img_size[0] / np.power(2, 3) + 2*pad)
        w = int(self.img_size[1] / np.power(2, 3) + 2*pad)
        patch_mask_relu4 = np.zeros((h - n + 1, w - n + 1, n*n), dtype=int)

        for i in range(h - n + 1):
            for j in range(w - n + 1):
                for a in range(n):
                    for b in range(n):
                        k = a * n + b
                        patch_mask_relu4[i, j, k] = (i + a) * w + j + b

        patch_mask_temp = np.reshape(patch_mask_relu4, newshape=[-1])
        patch_mask_segment_relu4 = []
        for i in range(self.batch_size):
            patch_mask_segment_relu4.append(patch_mask_temp + i * h * w)
        patch_mask_segment_relu4 = np.reshape(patch_mask_segment_relu4, newshape=[-1])

        return patch_mask_relu5, patch_mask_segment_relu5, patch_mask_relu4, patch_mask_segment_relu4
