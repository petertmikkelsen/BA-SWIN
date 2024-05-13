import os
from os.path import expanduser, basename, splitext
import numpy as np
import pandas as pd
import tensorflow as tf
import re

class TFRecordDataset:

    def __init__(self, input, orig_shape, batch_size, imgs_file, lbls_file):
        if isinstance(input, list) or isinstance(input, np.ndarray):
            self.input_filenames = np.array([expanduser(f.strip()) for f in input])
        else:
            self.input_filenames = np.array([expanduser(f.strip()) for f in open(expanduser(input), 'r').readlines()])
        self.batch_size = batch_size
        self.orig_shape = orig_shape
        self.imgs_file = imgs_file
        self.lbls_file = lbls_file
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