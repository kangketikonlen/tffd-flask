import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Dropout,
    MaxPooling2D,
    GlobalAveragePooling2D,
)
from tensorflow.keras.optimizers import Adam

BASE_PATH = os.path.abspath(os.path.dirname("."))
UPLOAD_PATH = os.path.join(BASE_PATH, "uploads/")
OUTPUT_PATH = os.path.join(BASE_PATH, "outputs/")
SAMPLES_PATH = os.path.join(BASE_PATH, "samples/")
AI_PATH = os.path.join(BASE_PATH, "ai/")

classes = []
batch_size = 60
IMG_HEIGHT = 224
IMG_WIDTH = 224
split = 0.2

datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=split,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="reflect",
)

datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255, validation_split=split
)

train_data_generator = datagen_train.flow_from_directory(
    batch_size=batch_size,
    directory=OUTPUT_PATH,
    shuffle=True,
    seed=40,
    subset="training",
    interpolation="bicubic",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
)

vald_data_generator = datagen_val.flow_from_directory(
    batch_size=batch_size,
    directory=OUTPUT_PATH,
    shuffle=True,
    seed=40,
    subset="validation",
    interpolation="bicubic",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
)


def start():
    check_images()
    return generators()


def check_images():
    for class_name in os.listdir(OUTPUT_PATH):
        class_path = os.path.join(OUTPUT_PATH, class_name)
        if os.path.isdir(class_path):
            No_of_images = len(os.listdir(class_path))
            print("Found {} images of {}".format(No_of_images, class_name))
            classes.append(class_name)
    classes.sort()

def generators():
    display_images(train_data_generator)
    vald_data_generator.reset()
    train_data_generator.reset()
    model = Sequential(
        [
            Conv2D(
                16,
                3,
                padding="same",
                activation="relu",
                input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
            ),
            MaxPooling2D(),
            Dropout(0.10),
            Conv2D(32, 3, padding="same", activation="relu"),
            MaxPooling2D(),
            Conv2D(64, 3, padding="same", activation="relu"),
            MaxPooling2D(),
            Conv2D(128, 3, padding="same", activation="relu"),
            MaxPooling2D(),
            Conv2D(256, 3, padding="same", activation="relu"),
            MaxPooling2D(),
            GlobalAveragePooling2D(),
            Dense(1024, activation="relu"),
            Dropout(0.10),
            Dense(len(classes), activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # model.summary()
    return train(model)


def display_images(data_generator, no=15):
    sample_training_images, labels = next(data_generator)

    plt.figure(figsize=[25, 25])

    # By default we're displaying 15 images, you can show more examples
    total_samples = sample_training_images[:no]

    cols = 5
    rows = np.floor(len(total_samples) / cols)

    for i, img in enumerate(total_samples, 1):

        plt.subplot(rows, cols, i)
        plt.imshow(img)

        # Converting One hot encoding labels to string labels and displaying it.
        class_name = classes[np.argmax(labels[i - 1])]
        plt.title(class_name)
        plt.axis("off")

def train(model):
    history = model.fit(
        train_data_generator,
        steps_per_epoch=train_data_generator.samples // batch_size,
        epochs=1,
        validation_data=vald_data_generator,
        validation_steps=vald_data_generator.samples // batch_size,
    )

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(acc))

    plt.plot(epochs, acc, "b", label="Training acc")
    plt.plot(epochs, val_acc, "r", label="Validation acc")
    plt.title("Training accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title("Training loss")
    plt.legend()
    plt.show()

    target = "gilang1.jpg"
    images_path = os.path.join(SAMPLES_PATH, target)
    img = cv2.imread(images_path)

    imgr = cv2.resize(img, (224, 224))
    imgrgb = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)
    final_format = np.array([imgrgb]).astype("float64") / 255.0
    pred = model.predict(final_format)
    index = np.argmax(pred[0])
    prob = np.max(pred[0])
    label = classes[index]
    plt.imshow(img[:, :, ::-1])
    plt.axis("off")

    ai_path = os.path.join(AI_PATH, "datasets.h5")
    model.save(ai_path)
    results = "{} {:.2f}%".format(label, prob * 100)
    return results
