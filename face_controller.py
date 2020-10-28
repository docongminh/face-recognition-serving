import datetime
import os
import time
import logging
import cv2
import json
import requests
import mimetypes
# from pdf2image import convert_from_bytes
import numpy as np
import base64
import utils
import config
import tensorflow as tf # tf 1.x
from serving_utils import ServingController

UPLOADED_DIR='./logs'
# Check if upload path exits
path_original = os.path.join(UPLOADED_DIR, datetime.date.today().isoformat())
utils.check_exists(path_original)
RESULT_DIR=''
DEBUG_FLAG=True
ALLOW_FORMAT = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif', 'ppg', 'pgm']

class FaceController():
    """
        Class define support get embedding vector from image
    """
    def __init__(self, model_name):
        
        self.upload_path = ''
        self.result = {}
        self.model_name = model_name
        self.call_serving()

    def call_serving(self):
        """
            exec initilizer serving model 
        """
        self.init_serving = ServingController(serving_host=config.TF_SERVING_HOST,
                                        model_name=self.model_name,
                                        signature_name=config.model_config[self.model_name]['signature_name'])

    def image2embedding(self, image, data_type):
        """
            extract embeding vector from face images
            :input: image
                    data type: options url | base 64 ...
            :output: Return Dict{
                'embedding': numpy array,
                'time_extract': Time extract embedding vector
                'total_time': Total time execute all process
            }
        """
        start_time = time.time()
        # Init name of image
        filename = ''
        finished_time = 0
        # get or decode image
        try:
            if data_type == 'url':
                img_name = image.split('/')[-1]
                filename = str(datetime.datetime.now().isoformat()) + '_' + img_name
                if utils.is_url_image(image):
                    try:
                        r = requests.get(image, allow_redirects=True)
                    except:
                        self.result['error'] = 'Failed to open the URL'
                        return self.result, filename, finished_time
            elif data_type == 'base64':
                filename = str(datetime.datetime.now().isoformat()) + '_base64.jpg'
        except:
            self.result['error'] = 'Bad data'
            return self.result, filename, finished_time, face_path

        # Path to the original image
        self.upload_path = os.path.join(path_original, filename)
        print("path data: ", self.upload_path)
        # Save original image
        try:
            with open(self.upload_path, "wb") as f:
                if data_type == "url":
                    f.write(r.content)
                else:
                    imgdata = base64.b64decode(image)
                    f.write(imgdata)
        except EnvironmentError:
            print('OSError: Too many open files: ' + self.upload_path)

        if filename.split('.')[-1].lower() not in ALLOW_FORMAT:
            filename = filename + '.jpg'
        # read & preprocessing input
        img_raw = cv2.imread(self.upload_path)
        if img_raw.shape[0] != config.image_size or img_raw.shape[1] != config.image_size:
            img_raw = cv2.resize(img_raw, (config.image_size, config.image_size))
        rgb_image = utils.preprocessing(img_raw)
        print("Time encode: ", time.time()-start_time)
        # get result model
        start_time_ = time.time()
        # conduct extract result
        embedding, time_extract_embed = self.init_serving.get_embedding(image=rgb_image)
        end_service_time = time.time() - start_time_
        self.result['embedding'] = embedding
        self.result['time_extract'] = time_extract_embed
        self.result['total_time'] = end_service_time

        return self.result