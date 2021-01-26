import os
import glob
import shutil

from random import shuffle
from settings import CUR_DIR
from utils.folder_file_manager import get_index_from_file_path, load_text, save_file


def split_dataset_darknet():

    img_dir = os.path.join(CUR_DIR, 'darknet', 'custom_data', 'images')
    image_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
    xml_paths = glob.glob(os.path.join(CUR_DIR, 'training_dataset', 'xml', '*.txt'))
    image_indices = []

    for image_path in image_paths:

        _, index = get_index_from_file_path(path=image_path)
        if index != "":
            image_indices.append(index)

    for xml_path in xml_paths:

        xml_name, xml_index = get_index_from_file_path(path=xml_path)
        if xml_index in image_indices:

            xml_content = load_text(filename=xml_path)
            xml_content = xml_content.replace("15", "0")
            new_file_path = os.path.join(CUR_DIR, 'darknet', 'custom_data', 'images', xml_name)
            save_file(content=xml_content, filename=new_file_path, method='w')

    shuffle(image_indices)
    train_index = int(0.8 * len(image_indices))
    training_indices, test_indices = image_indices[:train_index], image_indices[train_index:]

    for idx in training_indices:

        path = "custom_data/images/image_{}.jpg".format(idx) + "\n"
        save_file(content=path, filename=os.path.join(CUR_DIR, 'darknet', 'custom_data', 'train.txt'), method='a')

    for idx in test_indices:

        path = "custom_data/images/image_{}.jpg".format(idx) + "\n"
        save_file(content=path, filename=os.path.join(CUR_DIR, 'darknet', 'custom_data', 'test.txt'), method='a')


if __name__ == '__main__':

    split_dataset_darknet()
