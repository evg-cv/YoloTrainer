import numpy as np
import cv2
import pandas as pd

from settings import HANDWRITTEN_DIGITS_PATH, RESIZE_HANDWRITTEN_DIGITS_PATH


def reshape_arabic_digits():

    arabic_digits_train_file_path = HANDWRITTEN_DIGITS_PATH
    training_digits_images = pd.read_csv(arabic_digits_train_file_path, compression="zip", header=None)

    resize_images = []
    for i in range(10000):

        image_values = training_digits_images.loc[i]
        image_array = np.asarray(image_values)
        image_array = image_array.reshape(28, 28).astype('uint8')
        resize_img = cv2.resize(image_array, (32, 32), interpolation=cv2.INTER_CUBIC)
        # resize_img = np.reshape(resize_img, (-1, 1))
        resize_img = resize_img.ravel()
        # cv2.imshow("digit image", image_array)
        # cv2.waitKey()
        resize_images.append(resize_img)

    resize_arabic_digits_train_file_path = RESIZE_HANDWRITTEN_DIGITS_PATH
    df = pd.DataFrame(resize_images)
    df.to_csv(resize_arabic_digits_train_file_path, index=False, header=False)


if __name__ == '__main__':

    reshape_arabic_digits()
