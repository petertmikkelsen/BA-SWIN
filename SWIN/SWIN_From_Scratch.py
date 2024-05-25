from SWIN_Transformer import *
from Training_utils import *
from os.path import join, exists
from os import makedirs
from keras import layers
import tensorflow as tf
import keras

# MAC weird stuff
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

image_shape = (1194, 938)
output_shape = (224, 224)
batch_size = 10
fold_to_use = 0
num_epochs = 11
learning_rate = 0.00001


# Prepare the data
# Step 1: Convert our TFRecord dataset to a numpy array, with the correct labels from the cvs files.
dataset_path = 'Mini - SWIN/DK_dataset_1213'
if not exists('SWIN/ModelsAndResults/test'):
    makedirs('SWIN/ModelsAndResults/test')
model_dir = 'SWIN/ModelsAndResults/test/test_model_fold_%i' % fold_to_use
image_training_list = join(dataset_path, 'training_fold_%i.txt' % fold_to_use)
image_validation_list = join(dataset_path, 'validation_fold_%i.txt' % fold_to_use)
imgs = 'Mini - SWIN/NoisyLabels/available_image_data_DK1213_fixed.csv'
lbls = 'Mini - SWIN/NoisyLabels/DK_labels_1213_to_23_fixed.csv'

print('Using dataset: %s' % dataset_path)

# TF record datasets
training_dataset = TFRecordDataset(
    input=image_training_list, 
    orig_shape=image_shape, 
    output_shape=output_shape, 
    batch_size=batch_size, 
    imgs=imgs,
    lbls=lbls
)

validation_dataset = TFRecordDataset(
    input=image_validation_list, 
    orig_shape=image_shape, 
    output_shape=output_shape, 
    batch_size=batch_size, 
    imgs=imgs,
    lbls=lbls
)

# init model
model, start_from_epoch = init_model(model_dir, learning_rate=learning_rate)

callbacks = init_loggers(model_dir, validation_dataset)

history = model.fit(
    x=training_dataset.get_input_fn().repeat(),
    steps_per_epoch=len(training_dataset),
    callbacks=callbacks,
    batch_size=batch_size,
    epochs=num_epochs,
    verbose=1,
    initial_epoch=start_from_epoch
    #validation_data=validation_dataset,
)

#with open(join(model_dir, 'training_history_fold_%i.csv' % fold_to_use), 'w') as f:
#    pd.DataFrame(history.history).to_csv(f, index=False)
