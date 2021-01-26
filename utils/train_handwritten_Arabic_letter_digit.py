import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from scipy.ndimage import rotate
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dropout, Dense
from keras.models import model_from_yaml
from keras.callbacks import ModelCheckpoint
from settings import HANDWRITTEN_DIGITS_TESTING_IMAGE_PATH, HANDWRITTEN_DIGITS_TESTING_LABEL_PATH, \
    HANDWRITTEN_DIGITS_TRAINING_IMAGE_PATH, HANDWRITTEN_DIGITS_TRAINING_LABEL_PATH, \
    HANDWRITTEN_LETTERS_TESTING_IMAGE_PATH, HANDWRITTEN_LETTERS_TESTING_LABEL_PATH, \
    HANDWRITTEN_LETTERS_TRAINING_IMAGE_PATH, HANDWRITTEN_LETTERS_TRAINING_LABEL_PATH


def prepare_dataset():

    # ------------------------------preparing for training and testing dataset----------------------------------------

    letters_training_images_file_path = HANDWRITTEN_LETTERS_TRAINING_IMAGE_PATH
    letters_training_labels_file_path = HANDWRITTEN_LETTERS_TRAINING_LABEL_PATH
    # Testing letters images and labels files
    letters_testing_images_file_path = HANDWRITTEN_LETTERS_TESTING_IMAGE_PATH
    letters_testing_labels_file_path = HANDWRITTEN_LETTERS_TESTING_LABEL_PATH

    # Loading dataset into dataframes
    training_letters_images = pd.read_csv(letters_training_images_file_path, compression='zip', header=None)
    training_letters_labels = pd.read_csv(letters_training_labels_file_path, compression='zip', header=None)
    testing_letters_images = pd.read_csv(letters_testing_images_file_path, compression='zip', header=None)
    testing_letters_labels = pd.read_csv(letters_testing_labels_file_path, compression='zip', header=None)

    # print statistics about the dataset
    # print("There are %d training arabic letter images of 32x32 pixels." % training_letters_images.shape[0])
    # print("There are %d testing arabic letter images of 32x32 pixels." % testing_letters_images.shape[0])

    # print(training_letters_images.head())

    # Training digits images and labels files
    digits_training_images_file_path = HANDWRITTEN_DIGITS_TRAINING_IMAGE_PATH
    digits_training_labels_file_path = HANDWRITTEN_DIGITS_TRAINING_LABEL_PATH
    # Testing digits images and labels files
    digits_testing_images_file_path = HANDWRITTEN_DIGITS_TESTING_IMAGE_PATH
    digits_testing_labels_file_path = HANDWRITTEN_DIGITS_TESTING_LABEL_PATH

    # Loading dataset into dataframes
    training_digits_images = pd.read_csv(digits_training_images_file_path, compression='zip', header=None)
    training_digits_labels = pd.read_csv(digits_training_labels_file_path, compression='zip', header=None)
    testing_digits_images = pd.read_csv(digits_testing_images_file_path, compression='zip', header=None)
    testing_digits_labels = pd.read_csv(digits_testing_labels_file_path, compression='zip', header=None)

    # print statistics about the dataset
    # print("There are %d training arabic digit images of 28x28 pixels." % training_digits_images.shape[0])
    # print("There are %d testing arabic digit images of 28x28 pixels." % testing_digits_images.shape[0])

    convert_values_to_image(training_letters_images.loc[12], True)
    print(training_letters_labels.loc[12])

    training_digits_images_scaled = training_digits_images.values.astype('float32') / 255
    training_digits_labels = training_digits_labels.values.astype('int32')
    testing_digits_images_scaled = testing_digits_images.values.astype('float32') / 255
    testing_digits_labels = testing_digits_labels.values.astype('int32')

    # training_letters_images_scaled = training_letters_images.values.astype('float32') / 255
    # training_letters_labels = training_letters_labels.values.astype('int32') - 1
    # testing_letters_images_scaled = testing_letters_images.values.astype('float32') / 255
    # testing_letters_labels = testing_letters_labels.values.astype('int32') - 1

    # print("Training images of digits after scaling")
    # print(training_digits_images_scaled.shape)
    #
    # print(training_digits_images_scaled[0:5])
    #
    # print("Training images of letters after scaling")
    # print(training_letters_images_scaled.shape)
    # print(training_letters_images_scaled[0:5])

    # one hot encoding
    # number of classes = 10 (digits classes) + 28 (arabic alphabet classes)
    number_of_classes = 10
    # training_letters_labels_encoded = to_categorical(training_letters_labels, num_classes=number_of_classes)
    # testing_letters_labels_encoded = to_categorical(testing_letters_labels, num_classes=number_of_classes)
    training_digits_labels_encoded = to_categorical(training_digits_labels, num_classes=number_of_classes)
    testing_digits_labels_encoded = to_categorical(testing_digits_labels, num_classes=number_of_classes)

    # print(training_digits_labels_encoded)

    # reshape input digit images to 64x64x1
    training_digits_images_scaled = training_digits_images_scaled.reshape([-1, 32, 32, 1])
    testing_digits_images_scaled = testing_digits_images_scaled.reshape([-1, 32, 32, 1])

    # reshape input letter images to 64x64x1
    # training_letters_images_scaled = training_letters_images_scaled.reshape([-1, 32, 32, 1])
    # testing_letters_images_scaled = testing_letters_images_scaled.reshape([-1, 32, 32, 1])

    # print(training_digits_images_scaled.shape, training_digits_labels_encoded.shape, testing_digits_images_
    # scaled.shape,
    #       testing_digits_labels_encoded.shape)
    # print(training_letters_images_scaled.shape, training_letters_labels_encoded.shape,
    #       testing_letters_images_scaled.shape, testing_letters_labels_encoded.shape)

    # training_data_images = np.concat  enate((training_digits_images_scaled, training_letters_images_scaled), axis=0)
    # training_data_labels = np.concatenate((training_digits_labels_encoded, training_letters_labels_encoded), axis=0)
    # print("Total Training images are {} images of shape".format(training_data_images.shape[0]))
    # print(training_data_images.shape, training_data_labels.shape)

    # testing_data_images = np.concatenate((testing_digits_images_scaled, testing_letters_images_scaled), axis=0)
    # testing_data_labels = np.concatenate((testing_digits_labels_encoded, testing_letters_labels_encoded), axis=0)
    # print("Total Testing images are {} images of shape".format(testing_data_images.shape[0]))
    # print(testing_data_images.shape, testing_data_labels.shape)

    # ----------------------------------------------------------------------------------------------------------------

    # -------------------------------------training all of the possible models ---------------------------------------
    # model = create_model()
    # model.summary()
    #
    # plot_model(model, to_file="model.png", show_shapes=True)
    # display(IPythonImage('model.png'))
    #
    # seed = 7
    # np.random.seed(seed)
    #
    # # define the grid search parameters
    # optimizer = ['RMSprop', 'Adam', 'Adagrad', 'Nadam']
    # kernel_initializer = ['normal', 'uniform']
    # activation = ['relu', 'linear', 'tanh']
    #
    # param_grid = dict(optimizer=optimizer, kernel_initializer=kernel_initializer, activation=activation)
    #
    # # count number of different parameters values combinations
    # parameters_number = 1
    # for x in param_grid:
    #     parameters_number = parameters_number * len(param_grid[x])
    # print("Number of different parameter combinations = {}".format(parameters_number))
    #
    # epochs = 5
    # batch_size = 20  # 20 divides the training data samples
    #
    # # creating the models with different hyperparameters
    # for a, b, c in [(x, y, z) for x in optimizer for z in activation for y in kernel_initializer]:
    #     params = {'optimizer': a, 'kernel_initializer': b, 'activation': c}
    #     print(params)
    #     curr_model = create_model(a, b, c)
    #     curr_model.fit(training_data_images, training_data_labels,
    #                    validation_data=(testing_data_images, testing_data_labels),
    #                    epochs=epochs, batch_size=batch_size, verbose=1)
    #     print("=============================================================================")

    # ---------------------------------------------------------------------------------------------------------------

    # -------------------------training the best model---------------------------------------------------------------
    model = create_model(optimizer='Adam', kernel_initializer='uniform', activation='relu')

    epochs = 20
    batch_size = 20
    #
    checkpointer = ModelCheckpoint(filepath='/media/mensa/Data/Task/EgyALPR/utils/model/models'
                                            '/arabic_handwritten_model/weights.hdf5',
                                   verbose=1, save_best_only=True)

    history = model.fit(training_digits_images_scaled, training_digits_labels_encoded,
                        validation_data=(testing_digits_images_scaled, testing_digits_labels_encoded),
                        epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[checkpointer])
    #
    plot_loss_accuracy(history)

    # ----------------------------------------------------------------------------------------------------------------

    # -----------------------testing trained model--------------------------------------------------------------------

    # model.load_weights('/media/mensa/Data/Task/EgyALPR/utils/model/models/arabic_handwritten_model/weights.hdf5')

    # Final evaluation of the model
    # metrics = model.evaluate(testing_letters_images_scaled, testing_letters_labels_encoded, verbose=1)
    # print("Test Accuracy: {}".format(metrics[1]))
    # print("Test Loss: {}".format(metrics[0]))

    # ----------------------------------------------------------------------------------------------------------------

    # -----------------------saving trained model---------------------------------------------------------------------

    model_yaml = model.to_yaml()
    with open("model/models/arabic_handwritten_model/digit_model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)

    model.save_weights("digit_model.h5")
    print("Saved model to disk")

    # ----------------------------------------------------------------------------------------------------------------

    # ------------------------predicting labels with testing images---------------------------------------------------

    # load YAML and create model
    # yaml_file = open('/media/mensa/Data/Task/EgyALPR/utils/model/models/arabic_handwritten_model/model.yaml', 'r')
    # loaded_model_yaml = yaml_file.read()
    # yaml_file.close()
    # loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    # loaded_model.load_weights("/media/mensa/Data/Task/EgyALPR/utils/model/models/arabic_handwritten_model/model.h5")
    # print("Loaded model from disk")

    # compile the loaded model
    # loaded_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # y_pred, y_true = get_predicted_classes(loaded_model, testing_data_images, testing_data_labels)

    # ----------------------------------------------------------------------------------------------------------------


def get_predicted_classes(model, data, labels=None):

    image_predictions = model.predict(data)
    predicted_classes = np.argmax(image_predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)

    return predicted_classes, true_classes


def create_model(optimizer='adam', kernel_initializer='he_normal', activation='relu'):
    # create model
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=3, padding='same', input_shape=(32, 32, 1),
                     kernel_initializer=kernel_initializer, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(
        Conv2D(filters=32, kernel_size=3, padding='same', kernel_initializer=kernel_initializer, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(
        Conv2D(filters=64, kernel_size=3, padding='same', kernel_initializer=kernel_initializer, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=3, padding='same', kernel_initializer=kernel_initializer,
                     activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())

    # Fully connected final layer
    model.add(Dense(10, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
    return model


def convert_values_to_image(image_values, display=False):
    image_array = np.asarray(image_values)
    image_array = image_array.reshape(32, 32).astype('uint8')
    # The original dataset is reflected so we will flip it then rotate for a better view only.
    image_array = np.flip(image_array, 0)
    image_array = rotate(image_array, -90)
    new_image = Image.fromarray(image_array)
    if display:
        new_image.show()

    return new_image


def plot_loss_accuracy(history):

    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)


if __name__ == '__main__':
    prepare_dataset()
