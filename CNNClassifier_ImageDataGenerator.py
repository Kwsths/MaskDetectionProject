import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import random
from shutil import copyfile
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model


def split_data(source, training, validation, split_size):
    """
    splits whole dataset into train and validation on proportion 70% train - 30% validation
    :param source: path of the dataset
    :param training: destination path that training dataset will be stored
    :param validation: destination path that validation dataset will be stored
    :param split_size: number indicating split size(here 0.7)
    """
    files = os.listdir(source)
    print(len(files))
    num = len(files) * split_size
    train_files = random.sample(files, int(num))
    test_files = [f for f in files if f not in train_files]
    for t in tqdm(train_files):
        new_file = os.path.join(source, t)
        if os.path.getsize(new_file) > 0:
            copyfile(new_file, training + t)
        else:
            print("{} is zero length, so ignoring".format(t))
    for t in tqdm(test_files):
        new_file = os.path.join(source, t)
        if os.path.getsize(new_file) > 0:
            copyfile(new_file, validation + t)
        else:
            print("{} is zero length, so ignoring".format(t))


def print_confusion_matrix(y_test, y_pred, labels):
    """
    Prints a confusion matrix, as a heat-map.
    :param y_test: true labels for test images
    :param y_pred: predicted labels for test images
    :param labels: names of the labels, 1-without_mask, 0-with_mask
    :return: The confusion matrix as heat-map
    """
    confm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(confm, index=labels, columns=labels)

    ax = sns.heatmap(df_cm, cmap='Oranges', annot=True, fmt='g')
    return ax


def plot_keras_history(history):
    """
    :param history:
    :return:
    """
    # the history object gives the metrics keys.
    # we will store the metrics keys that are from the training sesion.
    metrics_names = [key for key in history.history.keys() if not key.startswith('val_')]

    for i, metric in enumerate(metrics_names):

        # getting the training values
        metric_train_values = history.history.get(metric, [])

        # getting the validation values
        metric_val_values = history.history.get("val_{}".format(metric), [])

        # As loss always exists as a metric we use it to find the
        epochs = range(1, len(metric_train_values) + 1)

        # leaving extra spaces to allign with the validation text
        training_text = "   Training {}: {:.5f}".format(metric,
                                                        metric_train_values[-1])

        # metric
        plt.figure(i, figsize=(12, 6))

        plt.plot(epochs,
                 metric_train_values,
                 'b',
                 label=training_text)

        # if we validation metric exists, then plot that as well
        if metric_val_values:
            validation_text = "Validation {}: {:.5f}".format(metric,
                                                             metric_val_values[-1])

            plt.plot(epochs,
                     metric_val_values,
                     'g',
                     label=validation_text)

        # add title, xlabel, ylabe, and legend
        plt.title('Model Metric: {}'.format(metric))
        plt.xlabel('Epochs')
        plt.ylabel(metric.title())
        plt.legend()

        plt.show()



def create_folders():
    """
    function that creates folders in order to store training and validation data
    :return:
    """
    # define the path of the folder that contains source images, before divided into train-validation
    mask = os.path.join('dataset//with_mask')
    no_mask = os.path.join('dataset//without_mask')
    # create the destination folders
    os.mkdir('training_data')
    os.mkdir('validation_data')
    os.mkdir('training_data//with_mask')
    os.mkdir('training_data//without_mask')
    os.mkdir('validation_data//with_mask')
    os.mkdir('validation_data//without_mask')

    # split dataset into train - validation
    split_data(mask, 'training_data//with_mask//', 'validation_data//with_mask//', .7)
    split_data(no_mask, 'training_data//without_mask//', 'validation_data//without_mask//', .7)

def create_model():
    """
    function that creates, compile and fit model
    :return:
    """
    # define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), input_shape=(300, 300, 1), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # create instances of image data generator for training and validation set
    train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                       zoom_range=[0.5, 1.5],
                                       brightness_range=[0.5, 1.5])
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # define folder for input data
    train_generator = train_datagen.flow_from_directory(
        'training_data//',
        color_mode='grayscale',
        target_size=(300, 300),
        batch_size=128,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        'validation_data//',
        color_mode='grayscale',
        target_size=(300, 300),
        batch_size=64,
        class_mode='binary'
    )

    # create callbacks for the model
    monitor = 'val_loss'
    model_fname = 'cnn_model.h5'

    es = EarlyStopping(monitor=monitor, patience=5, verbose=1, restore_best_weights=True)
    mc = ModelCheckpoint(filepath=model_fname, monitor=monitor, save_best_only=True, save_weights_only=True, verbose=1)

    # fit the model
    history = model.fit_generator(train_generator,
                                  epochs=15,
                                  validation_data=val_generator,
                                  callbacks=[es, mc])

    # save model
    model.save('cnn_model.h5')
    # plot accuracy and loss on train - validation data
    plot_keras_history(history)


def test_model():
    """
    function that make predictions on test dataset
    also prints confusion matrix, classification report and ROC curve
    :return:
    """
    model = load_model('cnn_model.h5')
    # time to test our model
    # define and instance of image data generator for test data
    test_data_gen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_data_gen.flow_from_directory(
        'test_data//',
        target_size=(300, 300),
        batch_size=32,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=False
    )
    # take labels of test data
    real_classes = test_generator.classes
    # define the names of labels
    names = test_generator.class_indices.keys()
    # do predictions on test data
    predictions = model.predict(test_generator, batch_size=10)

    # store results on a list
    predicted_classes = []
    for p in predictions:
        predicted_classes.append(1) if p > 0.5 else predicted_classes.append(0)

    # print classification report
    report = classification_report(real_classes, predicted_classes, target_names=names)
    print(report)
    # print confusion matrix
    print_confusion_matrix(real_classes, predicted_classes, names)

    # plot ROC curve
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    fpr, tpr, thresholds = roc_curve(real_classes, predicted_classes)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='{} (area = {:.3f})'.format('Keras', auc_score))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    create_folders()
    create_model()
    test_model()
