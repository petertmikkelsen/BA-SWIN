import os
from os.path import expanduser, basename, splitext
import numpy as np
import pandas as pd
import tensorflow as tf
import re
import os
from os.path import expanduser, join, splitext, basename, exists
from sklearn.metrics import roc_curve, roc_auc_score, log_loss, confusion_matrix, balanced_accuracy_score
from os import makedirs
from keras import layers
import glob
from SWIN_Transformer import *
import csv


#MAC weird stuff
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class TFRecordDataset:

    def __init__(self, input, orig_shape, batch_size, output_shape, imgs=None, lbls=None,
                augment=False, augment_level=3):
        if isinstance(input, list) or isinstance(input, np.ndarray):
            self.input_filenames = np.array([expanduser(f.strip()) for f in input])
        else:
            self.input_filenames = np.array([expanduser(f.strip()) for f in open(expanduser(input), 'r').readlines()])
        self.batch_size = batch_size
        self.orig_shape = orig_shape
        self.imgs_file = imgs
        self.lbls_file = lbls
        self.augment = augment
        self.augment_level = 3
        self.output_shape = output_shape
        self.target_pixel_mean = 0.0
        self.target_pixel_sd = 0.0
        self.source_pixel_mean = 0.0
        self.source_pixel_sd = 0.0
        self.normalize = False
        self.db = None


        if self.imgs_file:
            self.db = pd.read_csv(self.imgs_file, usecols=['accession', 'bname'])

        if self.lbls_file:
            self._read_labels()

    def _read_labels(self):
        labels = pd.read_csv(self.lbls_file, usecols=['accession', 'cancer_label'])
        labels = pd.merge(self.db, labels, on='accession')
        self.bname_to_label = dict(zip(labels.bname, labels.cancer_label))
        self.labels = np.array([self.bname_to_label[re.sub(r'(_PROC(.+)?$|^r|^p|_\d$)', '', splitext(basename(tfrecord_path))[0])] for tfrecord_path in self.input_filenames])
        self.binary_labels = (self.labels > 0).astype(np.uint8)

    def __len__(self):
        return int(np.ceil(len(self.input_filenames) / self.batch_size))
    
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

        dataset = dataset.map(swin_preprocess_image)

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

#Probably not an image actually    
def swin_preprocess_image(image, label, image_dimension=1280):
    image = tf.image.resize_with_pad(image, target_height=image_dimension, target_width=image_dimension)
    #image = tf.expand_dims(image, axis=-1)
    image = MyChannelRepeat(3)(image)
    #label = tf.one_hot(label, 2, dtype=tf.int32)
    image = tf.image.resize(image, (224, 224), method='bicubic')
    return image, label



def init_loggers(model_dir, validation_dataset, multi_flavor_val=False):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=join(model_dir, 'logs/scalars/'), profile_batch=0, update_freq=100
    )

    model_callback = tf.keras.callbacks.ModelCheckpoint(
        join(model_dir, 'models/weights.{epoch:03d}-{val_AUC:.4f}.weights.h5'),
        monitor='val_AUC', verbose=0, save_best_only=False,
        save_weights_only=True, mode='auto', save_freq='epoch'
    )

    validation_store_callback = MyCallbacks(validation_dataset, model_dir, multi_flavor_val=multi_flavor_val)

    return [validation_store_callback, tensorboard_callback, model_callback]

def calculate_test_metrics_and_store_probs(probabilities, test_dataset, model_path, test_name):

    test_file_dir = join(expanduser(model_path), 'testing')
    if not exists(test_file_dir):
        makedirs(test_file_dir)

    labels = test_dataset.labels

    res_df = pd.DataFrame({
        'bname': [splitext(basename(x))[0] for x in test_dataset.input_filenames],
        'score': probabilities
    })

    res_df.to_csv(join(test_file_dir, 'testing_probabilities_%s.csv' % test_name), index=False)

    binary_labels = test_dataset.binary_labels

    AUC_all = roc_auc_score(binary_labels, probabilities)

    if np.sum((labels == 1)) > 0:
        AUC_sdc = roc_auc_score((labels == 1).astype(np.uint8), probabilities)
    else:
        AUC_sdc = 0.0

    if np.sum((labels == 2)) > 0:
        AUC_ic = roc_auc_score((labels == 2).astype(np.uint8), probabilities)
    else:
        AUC_ic = 0.0

    if np.sum((labels == 3)) > 0:
        AUC_ltc = roc_auc_score((labels == 3).astype(np.uint8), probabilities)
    else:
        AUC_ltc = 0.0

    print('\n============ TESTING PERFORMANCE ============' \
          'AUC (all):   %.2f ---- AUC (SDC):   %.2f ---- AUC (IC):  %.2f   ---- AUC (LTC):       %.2f\n' \
          '=============================================' % (AUC_all, AUC_sdc, AUC_ic, AUC_ltc), '\n')

class MyCallbacks(tf.keras.callbacks.Callback):

    def __init__(self, val_dataset, model_path, multi_flavor_val=False):
        super().__init__()
        self.val_dataset = val_dataset
        self.model_path = model_path
        self.multi_flavor_val = multi_flavor_val

    def on_epoch_end(self, epoch, logs={}):

        print('\n\n', 'Validation for epoch %i starting: ' % (epoch + 1))
        probabilities = self.model.predict(
            x=self.val_dataset.get_input_fn(),
            steps=len(self.val_dataset),
            verbose=1
        ).ravel()

        val_file_dir = join(expanduser(self.model_path), 'validation')
        if not exists(val_file_dir):
            makedirs(val_file_dir)

        res_df = pd.DataFrame({
            'bname': [splitext(basename(x))[0] for x in self.val_dataset.input_filenames],
            'score': probabilities,
            'label': self.val_dataset.labels
        })

        if self.multi_flavor_val:
            res_df = res_df.rename(columns={'bname': 'flavor_bname'})
            res_df = res_df.assign(bname=res_df.flavor_bname.map(lambda x: re.sub(r'(_PROC(.+)?$|^r|^p)', '', x)))
            res_df = res_df[['bname', 'score', 'label']]
            res_df = res_df.groupby('bname', sort=False).agg({'score': 'mean', 'label': 'max'}).reset_index()

        res_df.to_csv(join(val_file_dir, 'validation_probabilities_E%d.csv' % (epoch + 1)), index=False)

        probabilities = res_df.score.to_numpy()
        labels = res_df.label.to_numpy()
        binary_labels = (labels > 0).astype(np.uint8)

        AUC_all = roc_auc_score(binary_labels, probabilities)
        BCE_all = log_loss(binary_labels, probabilities.astype(np.float64))

        if np.sum((labels == 1)) > 0:
            AUC_sdc = roc_auc_score((labels == 1).astype(np.uint8), probabilities)
            BCE_sdc = log_loss((labels == 1).astype(np.uint8), probabilities.astype(np.float64))
        else:
            AUC_sdc = 0.0
            BCE_sdc = 0.0

        if np.sum((labels == 2)) > 0:
            AUC_ic = roc_auc_score((labels == 2).astype(np.uint8), probabilities)
            BCE_ic = log_loss((labels == 2).astype(np.uint8), probabilities.astype(np.float64))
        else:
            AUC_ic = 0.0
            BCE_ic = 0.0

        if np.sum((labels == 3)) > 0:
            AUC_ltc = roc_auc_score((labels == 3).astype(np.uint8), probabilities)
            BCE_ltc = log_loss((labels == 3).astype(np.uint8), probabilities.astype(np.float64))
        else:
            AUC_ltc = 0.0
            BCE_ltc = 0.0

        fpr, tpr, thresholds = roc_curve(binary_labels, probabilities)
        optimal_threshold = thresholds[np.argmax(tpr - fpr)]

        predictions = (probabilities > optimal_threshold).astype(np.uint8)
        tn, fp, fn, tp = confusion_matrix(binary_labels, predictions.astype(np.float64)).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        bin_acc = balanced_accuracy_score(binary_labels, predictions.astype(np.float64))

        logs['val_AUC'] = tf.constant(AUC_all)
        logs['val_AUC_sdc'] = tf.constant(AUC_sdc)
        logs['val_AUC_ic'] = tf.constant(AUC_ic)
        logs['val_AUC_ltc'] = tf.constant(AUC_ltc)

        logs['val_loss'] = tf.constant(BCE_all)
        logs['val_loss_sdc'] = tf.constant(BCE_sdc)
        logs['val_loss_ic'] = tf.constant(BCE_ic)
        logs['val_loss_ltc'] = tf.constant(BCE_ltc)

        logs['val_thres'] = tf.constant(optimal_threshold)
        logs['val_sensitivity'] = tf.constant(sensitivity)
        logs['val_specificity'] = tf.constant(specificity)

        logs['val_binary_accuracy'] = tf.constant(bin_acc)

        def extract_float_values(logs):
            new_logs = {}
            for key, value in logs.items():
                if isinstance(value, tf.Tensor):
                    new_logs[key] = value.numpy().item()  # Convert tensor to a float value
                else:
                    new_logs[key] = value
            return new_logs
        
        csv_file_path = os.path.join(self.model_path, 'model_history.csv')

        file_exists = os.path.isfile(csv_file_path)

        logs = extract_float_values(logs)

        with open(csv_file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=logs.keys())

            if not file_exists:
                writer.writeheader()
            
            writer.writerow(logs)

        print('AUC (all):   %.2f ---- AUC (SDC):   %.2f ---- AUC (IC):  %.2f   ---- AUC (LTC):       %.2f\n'\
              'Loss (all):  %.2f ---- Loss (SDC):  %.2f ---- Loss (IC): %.2f   ---- Loss (LTC):      %.2f\n'\
              'Sensitivity: %.2f ---- Specificity: %.2f ---- Threshold: %.4f ---- Binary Accuracy: %.2f' % (
            logs['val_AUC'],
            logs['val_AUC_sdc'],
            logs['val_AUC_ic'],
            logs['val_AUC_ltc'],
            logs['val_loss'],
            logs['val_loss_sdc'],
            logs['val_loss_ic'],
            logs['val_loss_ltc'],
            logs['val_sensitivity'],
            logs['val_specificity'],
            logs['val_thres'],
            logs['val_binary_accuracy']
        ), '\n\n')

def init_model(model_dir, learning_rate):
    #SET INITIAL WEIGHTS
    model_weights_path = expanduser('/Users/jeppeaasteddue/DokumenterLokalt/BASWIN/Weights/weightsTiny.h5')
    start_from_epoch = 0

    if not exists(model_dir):
        print('Training from scratch.')
        makedirs(model_dir)
        makedirs(join(model_dir, 'models'))
    else:
        models = glob.glob(join(model_dir, 'models', '*'))
        if len(models) == 0:
            print('No models were stored. Training from scratch.')
        else:
            model_weights_path = max(glob.glob(join(model_dir, 'models', '*')), key=lambda x: int(re.findall(r'\.(\d{3})-', x)[0]))
            start_from_epoch = int(re.findall(r'\.(\d{3})-', model_weights_path)[0])
            print('Starting from checkpoint %s. Epoch=%i (one-indexed)' % (model_weights_path, start_from_epoch))

    #Model Creation

    model = build_swin_transformer((224, 224, 3), num_classes=1)

    print(model.summary())

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            #weight_decay=weight_decay
        ),
        metrics=[
            #keras.metrics.CategoricalAccuracy(name='accuracy'),
            #keras.metrics.TopKCategoricalAccuracy(k=5, name='top-5-accuracy'),
            tf.keras.metrics.AUC(curve='ROC', name='AUC'),
            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
            tf.keras.metrics.SensitivityAtSpecificity(specificity=0.85, name='SensAtSpec')
        ],
    )

   
    return model, start_from_epoch