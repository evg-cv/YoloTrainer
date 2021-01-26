import os
import time
import cv2
import numpy as np

from settings import LP_MODEL_DIR, LP_CONFIG_DIR, LP_NAMES_DIR


class LPDetector:

    def __init__(self):

        self.model_path = os.path.join(LP_MODEL_DIR, 'yolov2-tiny-custom_final.weights')
        self.config_path = os.path.join(LP_CONFIG_DIR, 'yolov2-tiny-custom.cfg')
        self.names_path = os.path.join(LP_NAMES_DIR, 'custom.names')

    def detect_lp_frame(self, frame_path):

        CONF_THRESH, NMS_THRESH = 0.5, 0.5

        # Load the network
        net = cv2.dnn.readNetFromDarknet(self.config_path, self.model_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Get the output layer from YOLO
        layers = net.getLayerNames()
        output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence
        # scores
        st_time = time.time()
        img = cv2.imread(frame_path)
        height, width = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layers)

        lp_frames = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONF_THRESH:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    lp_frame = img[y:y+h, x:x+w]
                    lp_frames.append(lp_frame)

                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        end_time = time.time()
        time_interval = end_time - st_time
        print("time elapsed:", time_interval)
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return lp_frames
