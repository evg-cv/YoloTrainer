import cv2
import numpy as np
from settings import ARABIC_LETTER_MODEL, ARABIC_LETTER_MODEL_YAML, ARABIC_DIGITS_MODEL, ARABIC_DIGITS_YAML
from keras.models import model_from_yaml


class LetterDetector:

    def __init__(self):

        self.letter_model = self.__load_model(yaml_path=ARABIC_LETTER_MODEL_YAML, model_path=ARABIC_LETTER_MODEL)
        self.digit_model = self.__load_model(yaml_path=ARABIC_DIGITS_YAML, model_path=ARABIC_DIGITS_MODEL)

    @staticmethod
    def __load_model(yaml_path, model_path):

        yaml_file = open(yaml_path, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        model.load_weights(model_path)
        print("Loaded model from disk")

        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

        return model

    def detect_letter_from_image(self, frame, side):

        # roi_gray_erosion = cv2.erode(roi_ gray_32, kernel=np.ones((2, 2), np.uint8), iterations=1)
        frame_trans = np.transpose(frame)
        # frame_trans = frame
        frame_reshape = frame_trans.reshape([-1, frame_trans.shape[0], frame_trans.shape[1], 1])
        frame_scaled = np.divide(frame_reshape, 255).astype('float32')

        y_pred = self.get_predicted_classes(data=frame_scaled, side=side)

        print(y_pred, side)
        cv2.imshow("roi image", frame_trans)
        cv2.waitKey(0)

        return y_pred

    def get_predicted_classes(self, data, side, labels=None):

        if side == "left":

            image_predictions = self.digit_model.predict(data)
        else:
            image_predictions = self.letter_model.predict(data)
        # res_msg = ""
        # for v in image_predictions[0]:
        #     res_msg += " " + str(round(v, 2))
        # print(res_msg)
        predicted_classes = np.argmax(image_predictions, axis=1)
        # true_classes = np.argmax(labels, axis=1)

        return predicted_classes


if __name__ == '__main__':

    detector = LetterDetector()

    roi_gray_erosion = cv2.imread("/test_with_h_margin/test_2.jpg",
                                  cv2.IMREAD_GRAYSCALE)

    # roi_gray_32 = cv2.resize(roi_gray_erosion, (32, 32), interpolation=cv2.INTER_CUBIC)
    img_flip_ud_lr = np.transpose(roi_gray_erosion)

    cv2.imshow("input", roi_gray_erosion)
    cv2.imshow("transpose", img_flip_ud_lr)
    cv2.waitKey(0)

    roi_gray = img_flip_ud_lr.reshape([-1, img_flip_ud_lr.shape[0], img_flip_ud_lr.shape[1], 1])
    roi_gray_scaled = np.divide(roi_gray, 255).astype('float32')
    res = detector.get_predicted_classes(data=roi_gray_scaled)
    print(res)
