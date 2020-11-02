from flask import Flask, jsonify, request
from face_controller import FaceController
import json
import base64

app = Flask(__name__)

def allowed_file(filename):
    """
    """
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict", methods=['POST'])
def get_predict():
    """
    	#
    """
    data = request.get_json()
    image = data["image"]
    data_type = data["data_type"]
    model_name = data["model_name"]
    face_controller = FaceController(model_name=model_name)
    result = face_controller.image2embedding(image=image, data_type=data_type)
    return data_type


if __name__ =='__main__':
    app.run(host="10.5.0.4", port=5000,debug=True)