from network.model import ModelNet
from network.ops import *
from utils import *

class Inference(object):
    def __init__(self, args):
        self.checkpoint_dir = args['checkpoint_dir']
        self.patch_size = args['patch_size']

        self.model = ModelNet(mode='test', args=args)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())

        dir_checkpoint = self.checkpoint_dir
        if os.path.isdir(dir_checkpoint):
            ckpt = tf.train.get_checkpoint_state(dir_checkpoint)
            if ckpt and ckpt.model_checkpoint_path:
                print("Restoring from checkpoint...")
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                vars = tf.trainable_variables()
                t_vars = [var for var in vars if 'decoder' in var.name or 'attn' in var.name]
                self.saver = tf.train.Saver(var_list=t_vars)
                self.saver.restore(self.sess, os.path.join(dir_checkpoint, ckpt_name))
                print(" [*] Load checkpoint SUCCESS")
            else:
                raise Exception("No checkpoint found...")
        else:
            raise Exception("No checkpoint_dir found...")

    def predict(self, Ic, Is):
        patch_mask_c_relu5, patch_mask_segment_c_relu5, patch_mask_c_relu4, patch_mask_segment_c_relu4 =self.get_patch_mask(Ic)
        patch_mask_s_relu5, patch_mask_segment_s_relu5, patch_mask_s_relu4, patch_mask_segment_s_relu4 =self.get_patch_mask(Is)
        fetches = {
                   'Ics': self.model.Ics
                   }

        start = time.time()
        results = self.sess.run(fetches, feed_dict={self.model.Ic: Ic, self.model.Is: Is,
                                                    self.model.patch_mask_c_relu5: patch_mask_c_relu5,
                                                    self.model.patch_mask_s_relu5: patch_mask_s_relu5,
                                                    self.model.patch_mask_c_relu4: patch_mask_c_relu4,
                                                    self.model.patch_mask_s_relu4: patch_mask_s_relu4,

                                                    self.model.patch_mask_segment_c_relu5: patch_mask_segment_c_relu5,
                                                    self.model.patch_mask_segment_c_relu4: patch_mask_segment_c_relu4
                                                    })
        end = time.time()
        print("Stylized in:", end-start)
        return results, end - start


    def get_patch_mask(self, Img):
        n = self.patch_size   # patch size
        pad = n - 1

        # relu_5
        h = int(Img.shape[1] / np.power(2, 4) + 2*pad)
        w = int(Img.shape[2] / np.power(2, 4) + 2*pad)
        patch_mask_relu5 = np.zeros((h - n + 1, w - n + 1, n*n), dtype=int)

        for i in range(h - n + 1):
            for j in range(w - n + 1):
                for a in range(n):
                    for b in range(n):
                        k = a * n + b
                        patch_mask_relu5[i, j, k] = (i + a) * w + j + b

        patch_mask_segment_relu5 = np.reshape(patch_mask_relu5, newshape=[-1])

        # relu_4
        h = int(Img.shape[1] / np.power(2, 3) + 2*pad)
        w = int(Img.shape[2] / np.power(2, 3) + 2*pad)
        patch_mask_relu4 = np.zeros((h - n + 1, w - n + 1, n*n), dtype=int)

        for i in range(h - n + 1):
            for j in range(w - n + 1):
                for a in range(n):
                    for b in range(n):
                        k = a * n + b
                        patch_mask_relu4[i, j, k] = (i + a) * w + j + b

        patch_mask_segment_relu4 = np.reshape(patch_mask_relu4, newshape=[-1])

        return patch_mask_relu5, patch_mask_segment_relu5, patch_mask_relu4, patch_mask_segment_relu4
