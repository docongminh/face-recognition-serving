# import neccesary lib
import tensorflow as tf

# serving config
TF_SERVING_HOST = '10.1.19.160:8500'
# TF_SERVING_HOST = '10.1.16.148:8500'

#
model_config = {
    'resnet_100':{
        'signature_name' : tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    },
    'resnet_50':{
        'signature_name' : tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    },
    'mobilefacenet':{
        'signature_name' : tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    }
}
_dir = './test_serving'

# face config
image_size = 112

# data test path
data_test = './test_serving'