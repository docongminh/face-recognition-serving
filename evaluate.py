import os
import cv2
import numpy as np
import utils
import tqdm
import config
import json
from serving_utils import ServingController
import tensorflow as tf


class Evaluate():
    def __init__(self,
            register_path,
            eval_data_path,
            model_name,
            database_dir,
            database_name
        ):
        self.register_path = register_path
        self.eval_data_path = eval_data_path
        self.model_name = model_name
        self.database_dir = database_dir
        self.database_name = database_name
        # init 
        self.call_serving()

    def call_serving(self):
        """
            exec initilizer serving model 
        """
        self.init_serving = ServingController(serving_host=config.TF_SERVING_HOST,
                                        model_name=self.model_name,
                                        signature_name=config.model_config[self.model_name]['signature_name'])

    def create_db_embedding(self):
        """

        """
        # config dataset
        utils.check_exists(self.database_dir)
        db_path = os.path.join(self.database_dir, self.database_name)
        # 
        list_data = os.listdir(self.register_path)
        list_meta = []
        for image in tqdm.tqdm(list_data):
            dict_meta = {}
            label = image.split(".")[0]
            full_path = os.path.join(self.register_path, image)
            image = utils.read_img(full_path)
            processed = utils.preprocessing(image)
            embedding, total_time = self.init_serving.get_embedding(processed)
            dict_meta['full_path'] = full_path
            dict_meta['label'] = label
            dict_meta['embedding'] = embedding.tolist()
            list_meta.append(dict_meta)

        with open(db_path, 'w') as f:
            json.dump(list_meta, f, ensure_ascii=False, indent=4)
        
        return list_meta
    
    def get_embeding_newface(self, new_face):
        """
            get embeding vector to verification
        """
        new_embedding, _ = self.init_serving.get_embedding(new_face)
        return new_embedding
    
    def test_compute_faces(self, face_1, face_2):
        """
        
        """
        face_encodings = utils.preprocess_output(self.get_embeding_newface(face_1))
        face_to_compare = utils.preprocess_output(self.get_embeding_newface(face_2))
        if len(face_encodings) == 0:
            return np.empty((0))    
        face_dist_value = np.linalg.norm([face_encodings] - face_to_compare, axis=1)
        print('[Face Services | face_distance] Distance between two faces is {}'.format(face_dist_value))
        return face_dist_value
       
if __name__ == '__main__':
    init_eval = Evaluate(register_path=config.path_register,
                        eval_data_path=config.path_test,
                        model_name='resnet_100',
                        database_dir='database',
                        database_name='example.json')
    # list_meta = init_eval.create_db_embedding()
    face_1 = cv2.imread('/home/minhdc/face_test/test_celeb/test/11/15.png')
    face_2 = cv2.imread('/home/minhdc/face_test/test_celeb/test/11/14.png')
    face_3 = cv2.imread('/home/minhdc/face_test/test_celeb/test/49/14.png')
    distance = init_eval.test_compute_faces(utils.preprocessing(face_1), utils.preprocessing(face_2))
    distance_ = init_eval.test_compute_faces(utils.preprocessing(face_1), utils.preprocessing(face_3))




