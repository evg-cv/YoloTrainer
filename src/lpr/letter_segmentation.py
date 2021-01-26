import cv2
import numpy as np

from src.lpr.letter_recognition import LetterDetector
from settings import LETTER_LP_RATIO


class LetterSegmentation:

    def __init__(self):

        self.letter_detector = LetterDetector()

    @staticmethod
    def optimize_crop(img):

        height = img.shape[0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]

        if len(areas) != 0:
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            x, y, w, h = cv2.boundingRect(cnt)
            second_crop = img[y:y + h, x:x + w]
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            second_crop = img

        # result_img = second_crop[int(LETTER_LP_RATIO * height):, :]
        result_img = second_crop
        cv2.imshow("crop image", second_crop)
        cv2.waitKey()

        return result_img

    @staticmethod
    def auto_canny(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged

    @staticmethod
    def get_contour(frame):

        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

        return sorted_contours

    @staticmethod
    def resize_frame(roi_frame):

        _roi_h, _roi_w = roi_frame.shape[:2]

        _a = max(_roi_h, _roi_w)
        _b = min(_roi_h, _roi_w)
        zeros_img = np.zeros((_a, _a), dtype=np.uint8) * 255
        if _roi_h > _roi_w:
            zeros_img[:, (_a - _b) // 2: (_a - _b) // 2 + _b] = roi_frame
        else:
            zeros_img[(_a - _b) // 2: (_a - _b) // 2 + _b, :] = roi_frame

        roi_gray_32 = cv2.resize(zeros_img, (32, 32), interpolation=cv2.INTER_CUBIC)

        return roi_gray_32

    def segment_character(self, frames):

        kernel = np.ones((3, 3), np.uint8)
        lp_numbers = []

        for frame in frames:

            lp_number = ""

            crop_frame = self.optimize_crop(img=frame)
            img_area = crop_frame.shape[0] * crop_frame.shape[1]

            gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("gray image", gray)
            # cv2.waitKey()

            thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 72)
            cv2.imshow("thresh image", thresh_inv)
            cv2.waitKey()

            dilate_frame = cv2.dilate(thresh_inv.copy(), kernel=kernel, iterations=1)
            # cv2.imshow("dilate image", dilate_frame)
            # cv2.waitKey()

            erosion_frame = cv2.erode(thresh_inv.copy(), kernel=kernel, iterations=1)
            # cv2.imshow("opening image", erosion_frame)
            # cv2.waitKey()

            ero_cnt = self.get_contour(frame=erosion_frame)
            cnt_len = len(ero_cnt)
            dil_cnt = []
            dil_frame = erosion_frame
            while cnt_len > 2:

                dil_frame = cv2.dilate(dil_frame, kernel=kernel, iterations=1)
                dil_cnt = self.get_contour(frame=dil_frame)
                cnt_len = len(dil_cnt)

            left_contour = dil_cnt[0]
            right_contour = dil_cnt[1]

            edges = self.auto_canny(image=dilate_frame)
            letter_contours = self.get_contour(frame=edges)

            candi_rois = []
            for ctr in letter_contours:

                x, y, w, h = cv2.boundingRect(ctr)
                roi_area = w * h
                roi_ratio = roi_area / img_area
                # print(roi_ratio)

                if 0.01 <= roi_ratio and 0.8 <= h / w < 3:
                    candi_rois.append([x, y, w, h])

            max_w, max_h = np.max(np.array(candi_rois), axis=0)[-2:]

            for i, roi in enumerate(candi_rois):

                x, y, w, h = roi

                left_side = cv2.pointPolygonTest(left_contour, (x, y), True)
                right_side = cv2.pointPolygonTest(right_contour, (x, y), True)

                if left_side > right_side:

                    letter_side = "left"
                else:
                    letter_side = "right"

                margin_h = int((max_h - h) * 0.35)
                margin_w = int((max_w - w) * 0.35)
                # roi_frame = thresh_inv[y-margin_h:y+h+margin_h, x-margin_w:x+w+margin_w]
                roi_frame = crop_frame[y - margin_h:y + h + margin_h, x - margin_w:x + w + margin_w]
                roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                _, roi_thresh = cv2.threshold(roi_gray, 127, 255, cv2.THRESH_BINARY_INV)
                # cv2.imshow("roi gray", roi_thresh)
                # cv2.waitKey()

                roi_resize_frame = self.resize_frame(roi_frame=roi_thresh)
                cv2.imwrite("test_{}.jpg".format(i), roi_resize_frame)

                # roi_resize_frame = cv2.erode(roi_resize_frame, kernel=np.ones((2, 2), np.uint8), iterations=1)
                # cv2.imshow("roi erode", roi_resize_frame)
                # cv2.waitKey()

                letter = self.letter_detector.detect_letter_from_image(frame=roi_resize_frame, side=letter_side)

                lp_number += str(letter[0])

            lp_numbers.append(lp_number)

        return lp_numbers


if __name__ == '__main__':

    import pandas as pd

    letters_testing_images_file_path = "/media/mensa/Data/Task/EgyALPR/data/ahcd1/csvTestImages 3360x1024.zip"

    testing_letters_images = pd.read_csv(letters_testing_images_file_path, compression='zip', header=None)
    testing_letters_images_scaled = testing_letters_images.values.astype('float32') / 255
    testing_letters_images_scaled = testing_letters_images_scaled.reshape([-1, 32, 32, 1])

    # digits_testing_images_file_path = "/media/mensa/Data/Task/EgyALPR/data/ahdd1/csvTestImages 10k x 1024.zip"
    # testing_digits_images = pd.read_csv(digits_testing_images_file_path, compression='zip', header=None)
    # testing_digits_images_scaled = testing_digits_images.values.astype('float32') / 255
    # testing_digits_images_scaled = testing_digits_images_scaled.reshape([-1, 32, 32, 1])
    test_img = (testing_letters_images_scaled[12] * 255).astype(np.uint8)
    cv2.imshow("test", test_img)
    cv2.waitKey(0)
    #
    # for i in range(56):
    #     print(i)
    #     test_img = (testing_letters_images_scaled[i] * 255).astype(np.uint8)
        # cv2.imwrite("/media/mensa/Data/Task/EgyALPR/data/letters/letter_{}.jpg".format(i // 2), test_img)
        # test_img = (testing_digits_images_scaled[i] * 255).astype(np.uint8)
        # cv2.imshow("test", test_img)
        # cv2.waitKey(0)

    # digits_testing_labels_file_path = "/media/mensa/Data/Task/EgyALPR/data/ahdd1/csvTestLabel 10k x 1.zip"
    # testing_digits_labels = pd.read_csv(digits_testing_labels_file_path, compression='zip', header=None)
    # testing_digits_labels = testing_digits_labels.values.astype('int32')

    y_pred = LetterDetector().get_predicted_classes(testing_letters_images_scaled[:10], side="right")
    # y_pred = detector.get_predicted_classes(testing_digits_images_scaled[:10])
    print(y_pred)
