import argparse
import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.callbacks import EarlyStopping

from imgsearch.dataset import Dataset
from imgsearch.plot_history import plot_history

if __name__ == '__main__':

    os.system('clear')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = argparse.ArgumentParser(description='Train a ResNet152-backbone neural network')

    parser.add_argument('-d', '--training_path', type=str, help='path to the training folder to use')
    parser.add_argument('-n', '--name', type=str, default='resnet', help='name of the model to train and save')
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

    # Initialize the ResNet152 backbone without classifier

    resnet = ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the first 4 convolutional blocks

    for layer in resnet.layers:
        if 'conv5' in layer.name:
            break
        layer.trainable = False

    # Create a custom classifier on top

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = resnet(inputs, training=False)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(10, activation='softmax', kernel_regularizer='l1_l2')(x)
    outputs = layers.Dropout(0.1, seed=999)(x)

    model = tf.keras.Model(inputs, outputs)

    # Compile and train

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=3)

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[early_stop])

    plot_history(history, args.name)

    # Remove the classifier

    features = model.get_layer('flatten').output

    model = tf.keras.Model(model.inputs, features)

    # Save the model to models/

    model.save(f'models/{args.name}')
