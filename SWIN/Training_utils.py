import os
from os.path import expanduser, basename, splitext
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from keras import layers

class TFRecordDataset:

    def __init__(self, input, orig_shape, batch_size, output_shape, imgs=None, lbls=None):
        if isinstance(input, list) or isinstance(input, np.ndarray):
            self.input_filenames = np.array([expanduser(f.strip()) for f in input])
        else:
            self.input_filenames = np.array([expanduser(f.strip()) for f in open(expanduser(input), 'r').readlines()])
        self.batch_size = batch_size
        self.orig_shape = orig_shape
        self.imgs_file = imgs
        self.lbls_file = lbls
        self.output_shape = output_shape
        self.target_pixel_mean = 0.0
        self.target_pixel_sd = 0.0
        self.source_pixel_mean = 0.0
        self.source_pixel_sd = 0.0
        self.normalize = False
        self._read_labels()

    def _read_labels(self):
        db = pd.read_csv(self.imgs_file, usecols=['accession', 'bname'])
        labels = pd.read_csv(self.lbls_file, usecols=['accession', 'cancer_label'])
        labels = pd.merge(db, labels, on='accession')
        self.bname_to_label = dict(zip(labels.bname, labels.cancer_label))
        self.labels = np.array([self.bname_to_label[re.sub('(_PROC(.+)?$|^r|^p|_\d$)', '', splitext(basename(tfrecord_path))[0])] for tfrecord_path in self.input_filenames])

    def __len__(self):
        return int(np.ceil(len(self.input_filenames) / self.batch_size))

    def get_dataset(self):
        concurrent_threads = tf.data.experimental.AUTOTUNE
        label_dataset = tf.data.Dataset.from_tensor_slices(self.labels)
        dataset = tf.data.TFRecordDataset(self.input_filenames, num_parallel_reads=concurrent_threads)
        dataset = tf.data.Dataset.zip((dataset, label_dataset))

        def parse_proto(example_proto, label):
            features = {
                'X': tf.io.FixedLenFeature((np.prod(self.orig_shape),), tf.dtypes.float32)
            }
            parsed_features = tf.io.parse_single_example(example_proto, features)
            parsed_features['X'] = tf.cast(tf.reshape(parsed_features['X'], (*self.orig_shape, 1)), dtype=tf.dtypes.float32)
            return parsed_features['X'], label

        dataset = dataset.map(parse_proto, num_parallel_calls=concurrent_threads)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=concurrent_threads)

        return dataset
    
    def get_numpy_array(self):
        dataset = self.get_dataset()
        return np.array(list(dataset.as_numpy_iterator()))
    
    def get_input_fn(self):

        concurrent_threads = tf.data.experimental.AUTOTUNE
        label_dataset = tf.data.Dataset.from_tensor_slices(self.binary_labels)
        dataset = tf.data.TFRecordDataset(self.input_filenames, num_parallel_reads=concurrent_threads)
        dataset = tf.data.Dataset.zip((dataset, label_dataset))

        def parse_proto_external_labels(example_proto, label):
            features = {
                'X': tf.io.FixedLenFeature((np.prod(self.orig_shape),), tf.dtypes.float32)
            }

            parsed_features = tf.io.parse_single_example(example_proto, features)

            parsed_features['X'] = tf.cast(tf.reshape(parsed_features['X'], (*self.orig_shape, 1)), dtype=tf.dtypes.float32)

            if self.normalize:
                background_mask = tf.where(parsed_features['X'] == 0, 0.0, 1.0)
                parsed_features['X'] = parsed_features['X'] - self.source_pixel_mean
                parsed_features['X'] = tf.multiply(parsed_features['X'], (self.target_pixel_sd / self.source_pixel_sd))
                parsed_features['X'] = parsed_features['X'] + self.target_pixel_mean
                parsed_features['X'] = tf.multiply(parsed_features['X'], background_mask)

            if self.augment:
                # 0 to augment_level-1
                random_num = tf.cast(tf.floor(tf.random.uniform((), minval=0, maxval=self.augment_level, dtype=tf.dtypes.float32)), dtype=tf.dtypes.int32)

                def rotate(im):
                    rot_angle = tf.random.normal((), mean=0.0, stddev=15.0)
                    return lambda: tf.keras.layers.RandomRotation(rot_angle * np.pi / 180.0, interpolation='bilinear')(im)
                def zoom(im):
                    scale = tf.random.uniform((), minval=-.2, maxval=.2)
                    return lambda: tf.image.crop_and_resize([im], [[0.0, 0.0, 1.0 - scale, 1.0 - scale]], [0], self.orig_shape, method='bilinear')[0]
                def shear(im):
                    shear_par = tf.random.uniform((), minval=-.15, maxval=.15)
                    transforms = [1.0, shear_par, -(tf.math.maximum(0.0, shear_par)) * im.shape[1], 0.0, 1.0, 0.0, 0.0, 0.0]
                    return lambda: tf.raw_ops.ImageProjectiveTransformV2(images=im, transforms=transforms, output_shape=tf.shape(im), interpolation='BILINEAR')

                aug_img = tf.switch_case(random_num, branch_fns={
                    0: rotate(parsed_features['X']),
                    1: zoom(parsed_features['X']),
                    2: shear(parsed_features['X'])
                })

                if aug_img.shape[0] != self.output_shape[0] or aug_img.shape[1] != self.output_shape[1]:
                    aug_img = tf.image.resize(aug_img, self.output_shape, method='bicubic')

                return aug_img, label

            else:

                if parsed_features['X'].shape[0] != self.output_shape[0] or parsed_features['X'].shape[1] != self.output_shape[1]:
                    parsed_features['X'] = tf.image.resize(parsed_features['X'], self.output_shape, method='bicubic')

                return parsed_features['X'], label

        dataset = dataset.map(parse_proto_external_labels, num_parallel_calls=concurrent_threads)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=concurrent_threads)

        return dataset

class MyChannelRepeat(tf.keras.layers.Layer):
    def __init__(self, repeats, **kwargs):
        self.repeats = int(repeats)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        super(MyChannelRepeat, self).__init__(**kwargs)

    def compute_output_shape(self, shape):
        return (shape[0], shape[1], shape[2], shape[3] * self.repeats)

    def call(self, x, mask=None):
        return tf.concat([x, x, x], axis=-1)

    def get_config(self):
        base_config = super(MyChannelRepeat, self).get_config()
        return base_config