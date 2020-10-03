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

# from src.common import UPLOADED_DIR, RESULT_DIR, CROPPED_DIR, FACE_DIR, DEBUG_FLAG, DEBUG_IMAGE
# from src.apis import setup_logging

# from src.controllers import classifier
import cropper
import cropper_object
import detector_text
import reader
import dlib
from post_processing import post_process_name
# from src.controllers import detecting_and_reading
# from src.controllers import face_detector

class FaceController():
    """
    
    """
    def __init__(self):
        
        self.upload_path = ''
        self.embedding = None
        
    
    def 