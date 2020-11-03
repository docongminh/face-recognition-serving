import os
import logging
import time
import base64
#
import requests
from multiprocessing import Queue
from threading import Thread
import config
from face_controller import FaceController

face_controller = FaceController(model_name='resnet_50')
list_faces = os.listdir(config._dir)
for face in list_faces:
	t1 = time.time()
	print("---------------------------------------")
	print("face name: ", face)
	try:
		t1 = time.time()
		full_path = os.path.join(config._dir, face)
		with open(full_path, "rb") as face_file:
			base64_image = base64.b64encode(face_file.read())
		data_type='base64'
		result = face_controller.image2embedding(image=base64_image, data_type=data_type)
		print(result)
		print("Total time process an image: ", time.time() - t1)
	except expression as identifier:
		continue