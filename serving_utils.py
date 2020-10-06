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

def get_embedding(tf_server, image, model_name):
    """
        request serving by GRPC
    """
    t1 = time.time()
    image = image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    MAX_MESSAGE_LENGTH = -1
    t1 = time.time()
    channel = grpc.insecure_channel(tf_server, options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                                        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
    stub = PredictionServiceStub(channel)
    # request
    request_embedding = predict_pb2.PredictRequest()
    request_embedding.model_spec.name = model_name
    # request_embedding.model_spec.signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY # tf 2x
    request_embedding.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    t11 = time.time()
    request_embedding.inputs['input'].CopyFrom( tf.contrib.util.make_tensor_proto(image))

    embedding_obj = stub.Predict.future(request_embedding, None)
    embedding = embedding_obj.result().outputs
    runtime = time.time() - t1

    return np.array(embedding['output0'].float_val), runtime
