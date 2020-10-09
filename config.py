# import neccesary lib
import tensorflow as tf

# serving config
TF_SERVING_HOST = '10.1.19.160:8500'
# TF_SERVING_HOST = '10.1.16.148:8500'

#
model_type = 'resent_50'
# model_type = 'resent_100'
# model_type = 'mobilefacenet'
signature_name_mobileFaceNet = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
signature_name_resnet_100 = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
signature_name_resnet_50 = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
_dir = '/home/minhdc/Documents/F-Face/Simple_keras/images/test'

# face config
image_size = 112

# evaluate path config
path_register = '/home/minhdc/face_test/test_celeb/register'
path_test = '/home/minhdc/face_test/test_celeb/test'
db_dir = './database'

