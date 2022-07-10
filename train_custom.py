import argparse
import os

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from imgsearch.dataset import Dataset
from imgsearch.plot_history import plot_history

if __name__ == '__main__':

    os.system('clear')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = argparse.ArgumentParser(description='Train a ResNet152-backbone neural network')

    parser.add_argument('-d', '--training_path', type=str, help='path to the training folder to use')
    parser.add_argument('-n', '--name', type=str, default='myconv', help='name of the model to train and save')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='epochs to use during training phase')

    args = parser.parse_args()

    for device in tf.config.list_physical_devices():
        print(f'[INFO] Device {device.name} available')

    training_dataset = Dataset(args.training_path)

    training_paths = training_dataset.get_data_paths()
    training_classes = training_dataset.get_data_classes()

    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.training_path,
        labels='inferred',
        validation_split=0.2,
        subset="training",
        color_mode="rgb",
        seed=99,
        shuffle=True,
        image_size=(224, 224),
        batch_size=16
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.training_path,
        labels='inferred',
        validation_split=0.2,
        subset="validation",
        color_mode="rgb",
        seed=99,
        shuffle=True,
        image_size=(224, 224),
        batch_size=16
    )

    if args.name in os.listdir('models'):
        print(f'[INFO] A model named {args.name} already exist.')
        exit()

    # Initialize a custom Neural Network

    model = Sequential([

        layers.InputLayer(input_shape=(224, 224, 3)),

        layers.Conv2D(8, (7, 7), strides=(3, 3), activation='relu', kernel_regularizer='l1_l2'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Conv2D(16, (3, 3), strides=(2, 2), activation='relu', kernel_regularizer='l1_l2'),
        layers.BatchNormalization(),
        layers.Conv2D(16, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer='l1_l2'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', kernel_regularizer='l1_l2'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer='l1_l2'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.AvgPool2D(2, 2),
        layers.Flatten(name='features'),

        layers.Dense(10, activation='softmax', kernel_regularizer='l1_l2'),
        layers.Dropout(0.2)

    ])

    # Compile and train

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_delta=0.1, patience=3)

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[reduce_lr, early_stop])

    plot_history(history, args.name)

    # Remove the classifier

    features = model.get_layer("features").output

    model = tf.keras.Model(model.inputs, features)

    model.summary()

    # Save the model to models/

    model.save(f'models/{args.name}')
