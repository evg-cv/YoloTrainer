import os

from src.lp_detector.lp_detection import LPDetector
from src.lpr.letter_segmentation import LetterSegmentation
from settings import CUR_DIR


if __name__ == '__main__':

    lp_frames = LPDetector().detect_lp_frame(frame_path=os.path.join(CUR_DIR,
                                                                     'training_dataset', 'images', 'image_258.jpg'))
    LetterSegmentation().segment_character(frames=lp_frames)
