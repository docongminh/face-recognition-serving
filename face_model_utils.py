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

def request_serving(tf_server, image, model_name):
    """
        request serving by GRPC
    """
    MAX_MESSAGE_LENGTH = -1
    channel = grpc.insecure_channel(tf_server, options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                                        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
    stub = PredictionServiceStub(channel)
    # request
    request_embedding = predict_pb2.PredictRequest()
    request_embedding.model_spec.name = model_name
    # request_embedding.model_spec.signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY # tf 2x
    request_embedding.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    request_embedding.inputs['input'].CopyFrom( tf.contrib.util.make_tensor_proto(image))

    embedding_obj = stub.Predict.future(request_embedding, None)
    embedding = embedding_obj.result().outputs

    return np.array(embedding['output0'].float_val)

def get_embedding(tf_serving, image, model_name):
    """
        get embedding from serving model
    """
    start_time = time.time()
    embedding_vector = request_serving(tf_server=tf_serving, image=image, model_name=model_name)
    end_time = time.time()
    runtime = end_time - start_time



    return embedding_vector, runtime 
