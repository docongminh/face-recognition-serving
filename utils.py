import numpy as np
import mimetypes

# Check if the url image is valid
def is_url_image(url):
    mimetype, _ = mimetypes.guess_type(url)
    return (mimetype and mimetype.startswith('image'))

def to_rgb(img):
    w, h, _ = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret