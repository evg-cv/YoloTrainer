# DarknetTrainer

## Overview

This project is to detect the license plate of the cars and recognize the plate number. According to the character of 
the project, this consists of two main parts. 

One is to detect the license plate of the cars. Using the open ALPR API, 
the license plates of the cars are not detected. Therefore, the training part to detect the license plate is necessary,
which can be solved by using the Yolo model. The training dataset with the detected license plate is made manually 
from the images and video frames.

The second is to recognize the Arabic digits and letters in the license plate. It can not be solved by OCR using the 
tesseract framework in the condition under which the open OCR APIs like Google Vision API can not be used. So in this step, 
to recognize the Arabic letters and digits, the training part with the train dataset using the Keras framework is needed.
Currently, the train dataset is Arabic handwritten digits and letters dataset on Kaggle. 

## Configuration

- darknet
    
    The source code to train the Yolo model with the train dataset.
    
    There is no custom_data directory,  file in initial framework.
    
    We should make custom_data directory and download weights and conv file.
    
- training_dataset
    
    Annotated dataset with images and xmls
    
- utils
    
    Several functions to prepare for training dataset

## Installation

- Environment

    Ubuntu 18.04, Python 3.6, GPU
- Dependency Installation

    ```
        pip3 install -r requirements.txt
    ```
- Download models

    * For the model to detect the license plate, download from and copy them in /utils/model/models/lp_detection_model 
    directory.
    * For the model to recognize the handwritten Arabic digits and letters, download from 
    https://drive.google.com/file/d/1_cP94KT68NWMxf2En_jhk-KGDz9NlsIu/view?usp=sharing and copy them in 
    /utils/model/models/arabic_handwritten_model directory.

## Execution

- Please set the several path(DETECT_FRAME_PATH, HANDWRITTEN_DIGITS_TRAINING_PATH, HANDWRITTEN_DIGITS_TESTING_PATH,
HANDWRITTEN_LETTERS_TRAINING_PATH, HANDWRITTEN_LETTERSS_TESTING_PATH, etc) in settings file.

- Please run the following command in terminal in this project directory.

    ```
        python3 main.py
    ```

## Appendix

If you want to train the Yolo model and the handwritten recognition model, please read the following instructions.

- Training Yolo model
    
    * Preparation for training Yolo model
    
        Please download the darknet framework from https://github.com/pjreddie/darknet, make custom_data directory in it, 
        make the files whose name are custom.names and detector.data referencing 
        https://blog.francium.tech/custom-object-training-and-detection-with-yolov3-darknet-and-opencv-41542f2ff44e.
        Also, in the custom_data directory, the test and train text file is needed. The process to make these files is 
        referred in the execution step. And insert labels directory and images directory which contains the xml and jpg
        files for training in the custom_data folder. 
        
        Then download yolov3.weights from https://pjreddie.com/darknet/yolo/ and 
        darkenet53.conv.74 from https://pjreddie.com/media/files/darknet53.conv.74 and copy them in darknet directory.
    
    * Train the model to detect license plate of the cars
        
        After preparing for training as mentioned above, configure yolov3.cfg file followed by Configurations part in 
        https://blog.francium.tech/custom-object-training-and-detection-with-yolov3-darknet-and-opencv-41542f2ff44e
        If using GPU in training, please follow the instructions:
        
            1. Edit Makefile GPU=1
            2. Edit Makefile CUDNN=1
            3. Leave Makefile OPENCV=0
            4. Install re-install CUDA using pip (my advice is to use version 10.1)
            5. Edit Makefile NVCC=/usr/local/cuda/bin/nvcc (failure to do this will result in a recipe for target 'obj/convolutional_kernels.o' error during compilation)
            6. Compile (or recompile if you previously compiled) the Darknet binary build via admin privileges -- SUDO
            
            Also, modify batch size of yolov3-custom.config in custom_data/cfg according to image size.
            Following reference site, we may set batch size 64, but in our case, there happened an out of memory issue 
            while training.
            After modifying it 32, we could perform train well.
            
        Then plesae run the following command in the terminal. 
        ```
            ./darknet detector train custom_data/detector.data custom_data/cfg/yolov3-custom.cfg darknet53.conv.74
        ```
        
        After finishing train, copy the trained model(.weights) to utils/model/models/lp_detection_model directory.
      
- Training the model to recognize the handwritten digits and letters
    
    * Preparation for training the handwritten Arabic letters and digits
    
        Please download the handwritten Arabic digits and letters dataset from Kaggle and copy them /data directory. 
        Since the image sizes of the handwritten digits(28 * 28) are different from the one of handwritten letters
        (32 * 32), the adjustment between their sizes is needed.
        
        The adjustment between handwritten letters and handwritten digits:
        
            Please set HANDWRITTEN_DIGITS_PATH, RESIZE_HANDWRITTEN_DIGITS_PATH variables in settings file and run the 
            following command in terminal.
            
            ```
                python3 utils/arabic_digit_reshape.py
            ```
         
            Then the reshaped handwritten digits dataset(32 * 32) is saved in data/ahdd1 with csv file format.
         
        Training the model to recognize the handwritten digits and letters:
            
            Please set HANDWRITTEN_DIGITS_TESTING_IMAGE_PATH, HANDWRITTEN_DIGITS_TESTING_LABEL_PATH, 
            HANDWRITTEN_DIGITS_TRAINING_IMAGE_PATH, HANDWRITTEN_DIGITS_TRAINING_LABEL_PATH, 
            HANDWRITTEN_LETTERS_TESTING_IMAGE_PATH, HANDWRITTEN_LETTERS_TESTING_LABEL_PATH, 
            HANDWRITTEN_LETTERS_TRAINING_IMAGE_PATH, HANDWRITTEN_LETTERS_TRAINING_LABEL_PATH variables in settings file 
            and run the following command in terminal
            
            ```
                python3 utils/train_handwritten_Arabic_letter_digit.py 
            ```

## Main reference site

    https://blog.francium.tech/custom-object-training-and-detection-with-yolov3-darknet-and-opencv-41542f2ff44e