import datetime
import os
import time
import logging

import cv2
import json
import requests
import mimetypes
from pdf2image import convert_from_bytes
import numpy as np
import base64
import hashlib
import config
import face_detection_utils

UPLOADED_DIR=''
RESULT_DIR=''
DEBUG_FLAG=True

ALLOW_FORMAT = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif', 'ppg', 'pgm']

# Check if the url image is valid
def is_url_image(url):
    mimetype, _ = mimetypes.guess_type(url)
    return (mimetype and mimetype.startswith('image'))

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

class FaceController():
    """

    """
    def __init__(self):
        
        self.upload_path = ''
        self.result = {}

    
    def image2embedding(self, image, data_type, tf_serving):
        """
            extract embeding vector from face images
        """
        start_time = time.time()
        # Init name of image
        filename = ''
        finished_time = 0
        face_path = ''
        try:
            if data_type == 'url':
                img_name = image.split('/')[-1]
                filename = str(datetime.datetime.now().isoformat()) + '_' + img_name
                if is_url_image(image):
                    try:
                        r = requests.get(image, allow_redirects=True)
                    except:
                        self.result['error'] = 'Failed to open the URL'
                        return self.result, hash_result, filename, finished_time, face_path
            else:
                filename = str(datetime.datetime.now().isoformat()) + '_base64.jpg'
        except:
            self.result['error'] = 'Bad data'
            return self.result, filename, finished_time, face_path
        #
        # Check if upload path exits
        path_original = os.path.join(UPLOADED_DIR, datetime.date.today().isoformat())
        if not os.path.exists(path_original):
            os.makedirs(path_original, exist_ok=True)
        # Path to the original image
        self.upload_path = os.path.join(path_original, filename)

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
        
        # Verify that the uploaded image is valid
        if valid:
            if filename.split('.')[-1].lower() not in ALLOW_FORMAT:
                filename = filename + '.jpg'
            img_raw = cv2.imread(self.upload_path)
            if img_raw.shape[0] != config.image_size or img_raw.shape[1] != config.image_size:
                img_raw = cv2.resize(img_raw, (config.image_size, config.image_size))
            rgb_image = to_rgb(img_raw)
        face = 
        # get result model 
        embedding, embed_time = face_model_utils.get_embedding(tf_serving=tf_serving, image=rgb_image)


        #
