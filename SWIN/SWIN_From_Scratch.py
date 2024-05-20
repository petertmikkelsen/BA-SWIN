from SWIN_Transformer import *
from Training_utils import *
from os.path import join, exists
from os import makedirs
from keras import layers
import tensorflow as tf
import keras


image_shape = (1280, 1280)
image_flat = image_shape[0] * image_shape[1]
# image_shape = (1194, 938)
batch_size = 10
num_epochs = 60
learning_rate = 0.00001
fold_to_use = 0

num_classes = 2
input_shape = (1280, 1280, 1)
#input_shape = (1194, 938, 1)
patch_size = (2, 2)  # 2-by-2 sized patches
dropout_rate = 0.03  # Dropout rate
num_heads = 8  # Attention heads
embed_dim = 64  # Embedding dimension
num_mlp = 256  # MLP layer size
patch_channels = patch_size[0] * patch_size[1] * 3

# Convert embedded patches to query, key, and values with a learnable additive
# value
qkv_bias = True
window_size = 2  # Size of attention window
shift_size = 1  # Size of shifting window
image_dimension = 1280  # Initial image size

num_patch_x = input_shape[0] // patch_size[0]
num_patch_y = input_shape[1] // patch_size[1]

learning_rate = 1e-3
batch_size = 10
num_epochs = 40
validation_split = 0.1
weight_decay = 0.0001
label_smoothing = 0.1

# Prepare the data
# Step 1: Convert our TFRecord dataset to a numpy array, with the correct labels from the cvs files.
dataset_path = 'Mini - SWIN/DK_dataset_1213'
if not exists('ModelsAndResults/test'):
    makedirs('ModelsAndResults/test')
model_dir = 'ModelsAndResults/test/test_model_fold_%i' % fold_to_use
image_training_list = join(dataset_path, 'training_fold_%i.txt' % fold_to_use)
image_validation_list = join(dataset_path, 'validation_fold_%i.txt' % fold_to_use)
imgs = 'Mini - SWIN/NoisyLabels/available_image_data_DK1213_fixed.csv'
lbls = 'Mini - SWIN/NoisyLabels/DK_labels_1213_to_23_fixed.csv'

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

# get input fn
training_dataset = training_dataset.get_input_fn()
validation_dataset = validation_dataset.get_input_fn()

training_dataset = training_dataset.map(swin_preprocess_image)
validation_dataset = validation_dataset.map(swin_preprocess_image)

training_dataset = training_dataset.map(lambda x, y: patch_extract(x, y))
validation_dataset = validation_dataset.map(lambda x, y: patch_extract(x, y))

# Step 2: Initialize the model and compile it

# TO-DO, check if all other channels are zero-padded due to just setting as 12 instead of my_channel_repeat
# input = MyChannelRepeat(3)(input)

# previous input to model
# image_input = tf.keras.Input(shape=image_shape + (1,), dtype=tf.float32)
# image_repeat = MyChannelRepeat(3)(image_input)

# input to model
# patch_x * patch_y * channels
input_layer = layers.Input(shape=(image_flat // 4, patch_channels))

# patches = PatchExtraction(patch_size, embed_dim)(input)
x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(input_layer)
x = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=0,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(x)

x = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=shift_size,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(x)

x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)
x = layers.GlobalAveragePooling1D()(x)
output = layers.Dense(num_classes, activation='softmax')(x)

# Initialize the model
model = tf.keras.Model(input_layer, output)

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    ),
    metrics=[
        keras.metrics.CategoricalAccuracy(name='accuracy'),
        keras.metrics.TopKCategoricalAccuracy(k=5, name='top-5-accuracy'),
    ],
)


history = model.fit(
    training_dataset,
    batch_size=batch_size,
    epochs=num_epochs,
    validation_data=validation_dataset,
)

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

"""
Let's display the final results of the training on CIFAR-100.
"""

loss, accuracy, top_5_accuracy = model.evaluate(validation_dataset)
print(f"Test loss: {round(loss, 2)}")
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")