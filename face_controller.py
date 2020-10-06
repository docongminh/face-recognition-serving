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
import serving_utils

UPLOADED_DIR=''
RESULT_DIR=''
DEBUG_FLAG=True

ALLOW_FORMAT = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif', 'ppg', 'pgm']


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
        img_raw = cv2.imread(self.upload_path)
        if img_raw.shape[0] != config.image_size or img_raw.shape[1] != config.image_size:
            img_raw = cv2.resize(img_raw, (config.image_size, config.image_size))
        # rgb_image = utils.to_rgb(img_raw)
        rgb_image = img_raw
        print("Time encode: ", time.time()-start_time)
        # get result model
        start_time_ = time.time()
        embedding, time_extract_embed = serving_utils.get_embedding(tf_server=tf_serving, image=rgb_image, model_name='mobileFaceNet')
        end_service_time = time.time() - start_time_
        self.result['embedding'] = embedding
        self.result['time_extract'] = time_extract_embed
        self.result['total_time'] = end_service_time

        return self.result
    
if __name__ == '__main__':
    face_controller = FaceController()
    list_faces = os.listdir(config._dir)
    for face in list_faces:
        print("---------------------------------------")
        print("face name: ", face)
        # try:
        t1 = time.time()
        full_path = os.path.join(config._dir, face)
        with open(full_path, "rb") as face_file:
            base64_image = base64.b64encode(face_file.read())
        data_type='base64'
        # print("time encode: ", time.time() - t1)
        result = face_controller.image2embedding(image=base64_image, data_type=data_type, tf_serving=config.TF_SERVING_HOST)
        print(result)