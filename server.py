from flask import Flask, jsonify, request
from face_controller import FaceController
import json
from PIL import Image
import base64
import io
import time
import os
import re

app = Flask(__name__)


def allowed_file(filename):
    """
    """
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict", methods=['POST'])
def get_predict():
    """#
    """
    data = request.get_json()
    image = data["image"]
    data_type = data["data_type"]
    model_name = data["model_name"]
    # print(image)
    base64_image = re.sub('^data:image/.+;base64,', '', image)
    t1 = time.time()
    face_controller = FaceController(model_name=model_name)
    print("couter timing: ", time.time() - t1)
    result = face_controller.image2embedding(image=base64_image, data_type=data_type)
    response = json.dumps(result, indent=4)
    return response


if __name__ =='__main__':
    app.run(host="10.5.0.4", port=5000,debug=True)