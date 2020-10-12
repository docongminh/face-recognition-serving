import numpy as np
import mimetypes
import sklearn
import cv2
import os

def check_exists(path):

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    else:
        print("{} existed".format(path))

def preprocess_output(embedding):

    return sklearn.preprocessing.normalize(embedding.reshape(1, -1)).flatten()
# Check if the url image is valid
def is_url_image(url):
    mimetype, _ = mimetypes.guess_type(url)
    return (mimetype and mimetype.startswith('image'))

def read_img(path):
    return cv2.imread(path)

def to_rgb(img):
    w, h, _ = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def preprocessing(img):
    """
    """
    img = cv2.resize(img, (112, 112))
    cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cvt_img