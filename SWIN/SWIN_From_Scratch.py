from SWIN_Transformer import SwinTransformer
from Training_utils import TFRecordDataset
from os.path import join, exists
from os import makedirs

image_shape = (1194, 938)
batch_size = 10
num_epochs = 60
learning_rate = 0.00001
fold_to_use = 0


# Prepare the data
# Step 1: Convert our TFRecord dataset to a numpy array, with the correct labels from the cvs files.

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
    input=image_training_list, 
    orig_shape=image_shape, 
    output_shape=image_shape, 
    batch_size=batch_size, 
    imgs=imgs,
    lbls=lbls
)

validation_dataset = TFRecordDataset(
    input=image_validation_list, 
    orig_shape=image_shape, 
    output_shape=image_shape, 
    batch_size=batch_size, 
    imgs=imgs,
    lbls=lbls
)


# Step 2: Initialize the model and compile it

