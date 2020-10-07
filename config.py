# import neccesary lib
import tensorflow as tf # tf 1.x

# serving config
# TF_SERVING_HOST = '10.1.19.160:8500'
TF_SERVING_HOST = '10.1.16.148:8500'
signature_name_mobileFaceNet = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
_dir = '/home/minhdc/Documents/F-Face/Simple_keras/images/test'
# face config
image_size = 112

