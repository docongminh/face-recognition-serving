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


class ServingController():
    """
        Init request serving model
    """
    request_embedding = None
    result = None
    finished_time = None
    stub = None

    def __init__(self, serving_host, model_name, signature_name,max_message_length=-1):
        """
            init variable
        """
        self.serving_host = serving_host
        self.model_name = model_name
        self. max_message_length = max_message_length
        self.signature_name = signature_name
        self.init_serving()
    
    def init_serving(self):
        """
            init spec serving for ServingController
        """
        channel = grpc.insecure_channel(self.serving_host, options=[('grpc.max_send_message_length', self.max_message_length),
                                                    ('grpc.max_receive_message_length', self.max_message_length)])
        self.stub = PredictionServiceStub(channel)
        # request
        self.request_embedding = predict_pb2.PredictRequest()
        self.request_embedding.model_spec.name = self.model_name
        # request_embedding.model_spec.signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY # tf 2x
        # self.request_embedding.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        self.request_embedding.model_spec.signature_name = self.signature_name

    def get_embedding(self, image):
        """
            request serving to conduct predict by GRPC
        """
        start_time = time.time()
        image = image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        self.request_embedding.inputs['input'].CopyFrom( tf.contrib.util.make_tensor_proto(image))
        embedding_obj = self.stub.Predict.future(self.request_embedding, None)
        embedding = embedding_obj.result().outputs
        self.finished_time = time.time() - start_time
        self.result = np.array(embedding['output0'].float_val)

        return self.result, self.finished_time
