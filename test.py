import numpy as np
import os
import cv2
import config
import tensorflow as tf
import tensorflow_serving
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub
import grpc
import time


def process_img(img_path):
    """
    
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (112, 112))
    image = np.expand_dims(img, axis=0)
    # print(image_raw_np_expanded.shape)
    image = image.astype(np.float32)
    # print(img)
    return image

MAX_MESSAGE_LENGTH = -1
tf_server = config.TF_SERVING_HOST
channel = grpc.insecure_channel(tf_server, options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
stub = PredictionServiceStub(channel)
# request
request_embedding = predict_pb2.PredictRequest()
print("---------------mobilefacenet-------------")
for img_ in os.listdir(config.data_test):
    t1 = time.time()
    full_path = os.path.join(config.data_test, img_)
    image = process_img(full_path)
    request_embedding.model_spec.name = 'mobilefacenet'
    # request_embedding.model_spec.signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY # tf 2x
    request_embedding.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY


    t1 = time.time()
    request_embedding.inputs['input'].CopyFrom( tf.contrib.util.make_tensor_proto(image))

    embedding_obj = stub.Predict.future(request_embedding, None)
    embedding = embedding_obj.result().outputs
    # print(list(embedding['output0'].float_val))
    print(len(list(embedding['output0'].float_val)))
    print("mobilefacenet time: ", time.time() - t1)
# # Arface resnet 100
# print("-------------arcface-----------")
# for img_ in os.listdir(dataset):
#     t11 = time.time()
#     full_path = os.path.join(dataset, img_)
#     image = process_img(full_path)
#     request_embedding.model_spec.name = 'arcface'
#     # request_embedding.model_spec.signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY # tf 2x
#     request_embedding.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY


#     t1 = time.time()
#     request_embedding.inputs['input'].CopyFrom( tf.contrib.util.make_tensor_proto(image))

#     embedding_obj = stub.Predict.future(request_embedding, None)
#     embedding = embedding_obj.result().outputs
#     # print(list(embedding['output0'].float_val))
#     print(len(list(embedding['output0'].float_val)))
#     print("arcface time: ", time.time() - t11)
# # Arface resnet 100