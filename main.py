import os
import logging
import time
import base64
#
from multiprocessing import Queue
from threading import Thread
import config


def main():
	# TODO
	list_faces = os.listdir(config._dir)
	for face in list_faces:
		print("---------------------------------------")
		print("face name: ", face)
		try:
			full_path = os.path.join(config._dir, face)
			with open(full_path, "rb") as face_file:
				base64_image = base64.b64encode(face_file.read())
			data_type='base64'
			username=''
			#TODO 
			# call face controller
		except expression as identifier:
			continue


if __name__ == '__main__':
	main()
