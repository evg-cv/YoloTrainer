import os

from utils.folder_file_manager import make_directory_if_not_exists

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
LP_MODEL_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'utils', 'model', 'models', 'lp_detection_model'))
ARABIC_LETTER_MODEL_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'utils', 'model', 'models',
                                                                    'arabic_handwritten_model'))
LP_CONFIG_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'utils', 'model', 'cfg'))
LP_NAMES_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'utils', 'model', 'names'))

ARABIC_LETTER_MODEL_WEIGHTS = os.path.join(ARABIC_LETTER_MODEL_DIR, 'weights.hdf5')
ARABIC_LETTER_MODEL = os.path.join(ARABIC_LETTER_MODEL_DIR, 'letter_model.h5')
ARABIC_LETTER_MODEL_YAML = os.path.join(ARABIC_LETTER_MODEL_DIR, 'letter_model.yaml')
ARABIC_DIGITS_MODEL = os.path.join(ARABIC_LETTER_MODEL_DIR, 'digit_model.h5')
ARABIC_DIGITS_YAML = os.path.join(ARABIC_LETTER_MODEL_DIR, 'digit_model.yaml')

DETECT_FRAME_PATH = ""
HANDWRITTEN_DIGITS_PATH = "/media/mensa/Data/Task/EgyALPR/data/ahdd1/csvTestImages 10k x 784.zip"
RESIZE_HANDWRITTEN_DIGITS_PATH = "/media/mensa/Data/Task/EgyALPR/data/ahdd1/csvTestImages 10k x 1024.csv"
HANDWRITTEN_DIGITS_TRAINING_IMAGE_PATH = "/media/mensa/Data/Task/EgyALPR/data/ahdd1/csvTrainImages 60k x 1024.zip"
HANDWRITTEN_DIGITS_TRAINING_LABEL_PATH = "/media/mensa/Data/Task/EgyALPR/data/ahdd1/csvTrainLabel 60k x 1.zip"
HANDWRITTEN_DIGITS_TESTING_IMAGE_PATH = "/media/mensa/Data/Task/EgyALPR/data/ahdd1/csvTestImages 10k x 1024.zip"
HANDWRITTEN_DIGITS_TESTING_LABEL_PATH = "/media/mensa/Data/Task/EgyALPR/data/ahdd1/csvTestLabel 10k x 1.zip"
HANDWRITTEN_LETTERS_TRAINING_IMAGE_PATH = "/media/mensa/Data/Task/EgyALPR/data/ahcd1/csvTrainImages 13440x1024.zip"
HANDWRITTEN_LETTERS_TRAINING_LABEL_PATH = "/media/mensa/Data/Task/EgyALPR/data/ahcd1/csvTrainLabel 13440x1.zip"
HANDWRITTEN_LETTERS_TESTING_IMAGE_PATH = "/media/mensa/Data/Task/EgyALPR/data/ahcd1/csvTestImages 3360x1024.zip"
HANDWRITTEN_LETTERS_TESTING_LABEL_PATH = "/media/mensa/Data/Task/EgyALPR/data/ahcd1/csvTestLabel 3360x1.zip"

LETTER_LP_RATIO = 0.35
