import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from PIL import ImageFile
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from PIL import Image

import os

train_dirs = ['data/no-watermark/', 'data/watermark/']

# credit to a random chinese blog for explaining how to check the data--there is some strange corruption 
# involved in transfer that breaks the model. 
def is_valid_jpg(jpg_file):
    with open(jpg_file, 'rb') as f:
        f.seek(-2, 2)
        buf = f.read()
        f.close()
        return buf ==  b'\xff\xd9'  
def clean_data(train_dir):
    tot_removed = 0
    for file in os.listdir(train_dir):
        if os.path.splitext(file)[1].lower() == '.jpeg':
            if not is_valid_jpg(train_dir + file):
                os.remove(train_dir + file)
                tot_removed += 1

    print('\nincomplete file : %d' % tot_removed)

[clean_data(x) for x in train_dirs]

def read_data(data_dir, batch_size, img_height,img_width, ):
    if not os.path.exists("test"):
        os.mkdir("test")
        os.mkdir("test/no-watermark")
        os.mkdir("test/watermark")
    else:
        for d in os.listdir("test"):
            for file in os.listdir("test/" + d):
                 os.rename("test/" + d + "/" + file, data_dir + d + "/" + file)

    for d in os.listdir(data_dir):
        files = os.listdir(data_dir + d)
        np.random.shuffle(files)
        testing_files = files[:int(.2 * len(files))]
        for file in testing_files:
            os.rename(data_dir + d + "/" + file, "test/" + d + "/" + file)



    # This code block is used from Tensorflow's official guide on reading in image data
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed= 123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "test",
    image_size=(img_height, img_width),
    batch_size=batch_size)
    return train_ds, val_ds, test_ds

train_ds,val_ds,test_ds = read_data("data/", 100, 180,180) 

def display_sample_data(train_ds):
    class_names = train_ds.class_names
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

def make_model():

    num_classes = 2
    model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
    ])

    model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
    return model


physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  assert tf.config.experimental.get_memory_growth(physical_devices[0])
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


model = make_model()
history = model.fit(train_ds,validation_data=val_ds,epochs=5)
result = model.evaluate(test_ds)
print(dict(zip(model.metrics_names, result)))
model.save("trained model")