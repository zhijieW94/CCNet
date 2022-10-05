import tensorflow as tf
from scipy import misc
import numpy as np
import random

class ImageData:
    def __init__(self, img_h, img_w, channels):
        self.img_h = img_h
        self.img_w = img_w
        self.channels = channels

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)

        # resize_h = tf.cond(tf.less(tf.shape(x_decode)[0], self.img_h), lambda: self.img_h, lambda: tf.shape(x_decode)[0])
        # resize_w = tf.cond(tf.less(tf.shape(x_decode)[1], self.img_w), lambda: self.img_w, lambda: tf.shape(x_decode)[1])

        resize_shape = (512,512)
        img = tf.image.resize_images(x_decode,resize_shape)
        img = tf.random_crop(img, [self.img_h, self.img_w,self.channels])
        img = tf.cast(img, tf.float32) / 255
        return img, filename


#save image
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0] // (size[1]), w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image
    return img

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    images = np.uint8(np.clip(images, 0, 1) * 255)
    # images = np.clip(images, 0, 255).astype(np.uint8)
    return misc.imsave(path, merge(images, size))

def load_data_testing(image_path, img_size=[512,512,3], data_process_type=None):
    img = misc.imread(image_path, mode='RGB')
    w, h, c = img.shape
    w = w - w%32
    h = h - h%32

    img = img[0:w, 0:h] #misc.imresize(img,(w,h))

    if data_process_type == 'crop':
        h, w = img.shape[0], img.shape[1]
        hc, wc = img_size[0], img_size[1]
        if h < hc or w < wc:  # Upscale to size if one side is too small
            size = hc if hc>wc else wc
            img = resize_to(img, resize=size)
            h, w = img.shape[0], img.shape[1]

        h_off = (h - hc) // 2
        w_off = (w - wc) // 2
        img = img[h_off:h_off + hc, w_off:w_off + wc]
    elif data_process_type == 'resize':
        img = misc.imresize(img, (img_size[0], img_size[1]))

    img = np.expand_dims(img, axis=0)
    img = img / 255
    return img

def resize_to(img, resize=512):
    '''Resize short side to target size and preserve aspect ratio'''
    height, width = img.shape[0], img.shape[1]
    if height < width:
        ratio = height / resize
        long_side = round(width / ratio)
        resize_shape = (resize, long_side, 3)
    else:
        ratio = width / resize
        long_side = round(height / ratio)
        resize_shape = (long_side, resize, 3)

    return misc.imresize(img, resize_shape)

