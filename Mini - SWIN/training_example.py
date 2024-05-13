from training_utils import *
from os.path import join
import sys
import pandas as pd
import types
import tensorflow as tf

#if len(sys.argv) < 2:
#    exit('Missing fold to use argument')

# Training params
image_shape = (1194, 938)
batch_size = 10
num_epochs = 60
shuffle_on_init = True
augment = False
augment_level = 3
learning_rate = 0.00001
fold_to_use = 0

print('Training fold %i' % fold_to_use)

# Dataset inputs
dataset_path = 'DK_dataset_1213'
if not exists('ModelsAndResults/test'):
    makedirs('ModelsAndResults/test')
model_dir = 'ModelsAndResults/test/test_model_fold_%i' % fold_to_use
image_training_list = join(dataset_path, 'training_fold_%i.txt' % fold_to_use)
image_validation_list = join(dataset_path, 'validation_fold_%i.txt' % fold_to_use)
imgs = 'NoisyLabels/available_image_data_DK1213_fixed.csv'
lbls = 'NoisyLabels/DK_labels_1213_to_23_fixed.csv'

print('Using dataset: %s' % dataset_path)

# TF record datasets
training_dataset = TFRecordDataset(
    input=image_training_list, orig_shape=image_shape, output_shape=image_shape,
    shuffle=shuffle_on_init, batch_size=batch_size, augment=augment,
    augment_level=augment_level, random_seed=fold_to_use, imgs=imgs,
    lbls=lbls
)

validation_dataset = TFRecordDataset(
    input=image_validation_list, orig_shape=image_shape, output_shape=image_shape,
    shuffle=False, batch_size=batch_size, augment=False,
    augment_level=0, random_seed=fold_to_use, imgs=imgs,
    lbls=lbls
)

# Setting up model and compile
# model, initial_epoch = init_model(model_dir, image_shape, learning_rate)

model, initial_epoch = init_model_SWIN(model_dir, image_shape, learning_rate)

# Setting up loggers
callbacks = init_loggers(model_dir, validation_dataset)

# to check to train_data type
train_data = training_dataset.get_input_fn()
#train_data = training_dataset.get_input_fn().repeat()

# Without np.array
# train_data = training_dataset.get_input_fn().as_numpy_iterator()
# But with just .numpy
# train_data = training_dataset.get_input_fn().numpy()


# with np.array
# train_data = np.array(list(training_dataset.get_input_fn().as_numpy_iterator()))

# for features, labels in train_data.take(1):
#     print("Feature batch shape:", features.shape)
#     print("Label batch shape:", labels.shape)
#     print("Feature batch dtype:", features.dtype)
#     print("Label batch dtype:", labels.dtype)

# np_train_data = np.asarray(train_data)

steps_per_epoch = len(training_dataset) // batch_size
if (len(training_dataset) % batch_size) != 0:
    steps_per_epoch += 1

#tf.random.set_seed(1)
#tf.config.run_functions_eagerly(True)

# Train the model
history = model.fit(
    x=train_data,
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    callbacks=callbacks,
    initial_epoch=initial_epoch,
    verbose=1
)

# Store training history
with open(join(model_dir, 'training_history_fold_%i.csv' % fold_to_use), 'w') as f:
    pd.DataFrame(history.history).to_csv(f, index=False)

print('Training done!')