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
            model_type,
            signature_name,
            database_dir,
            database_name,

        
        ):
        self.register_path = register_path
        self.eval_data_path = eval_data_path
        self.model_type = model_type
        self.signature_name = signature_name
        self.database_dir = database_dir
        self.database_name = database_name
        self.init_serving = ServingController(serving_host=config.TF_SERVING_HOST,
                                            model_name=self.model_type,
                                            signature_name=self.signature_name)
        self.create_db_embedding()

    def create_db_embedding(self):
        """

        """
        # config dataset
        utils.check_exists(self.database_dir)
        db_path = os.path.join(self.database_dir, self.database_name)
        # 
        list_data = os.listdir(self.database_dir)
        list_meta = []
        for image in tqdm.tqdm(list_data):
            dict_meta = {}
            label = image.split(".")[0]
            full_path = os.path.join(self.database_dir, image)
            image = utils.read_img(full_path)
            processed = utils.preprocessing(image)
            embedding, total_time = self.init_serving.get_embedding(processed)
            dict_meta['full_path'] = full_path
            dict_meta['label'] = label
            dict_meta['embedding'] = embedding.tolist()
            list_meta.append(dict_meta)

        with open(db_path, 'w') as f:
            json.dump(list_meta, f, ensure_ascii=False, indent=4)
    

if __name__ == '__main__':
    init_eval = Evaluate(register_path=config.path_register,
                        eval_data_path=config.path_test,
                        model_type='resnet_100',
                        signature_name=config.signature_name_resnet_100,
                        database_dir='database',
                        database_name='example.json')
    init_eval.create_db_embedding()




