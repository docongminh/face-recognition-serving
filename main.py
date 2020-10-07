import os
import logging
import time
import base64
#
from multiprocessing import Queue
from threading import Thread
import config
from face_controller import FaceController

face_controller = FaceController()
list_faces = os.listdir(config._dir)
for face in list_faces:
	t1 = time.time()
	print("---------------------------------------")
	print("face name: ", face)
	# try:
	t1 = time.time()
	full_path = os.path.join(config._dir, face)
	with open(full_path, "rb") as face_file:
		base64_image = base64.b64encode(face_file.read())
	data_type='base64'
	# print("time encode: ", time.time() - t1)
	result = face_controller.image2embedding(image=base64_image, data_type=data_type, tf_serving=config.TF_SERVING_HOST)
	print(result)
	print("Total time: ", time.time() - t1)

		# call face controller
	# except expression as identifier:
	# 	continue