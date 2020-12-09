from flask import Flask, render_template, request, jsonify
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import requests
from io import BytesIO
import json

app = Flask(__name__)
global blood_label;


@app.route('/', methods=['GET', 'POST'])
def home():

    url = request.data
    url = json.loads(url)['url']
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    prototxtPath = os.path.sep.join(['models', "deploy.prototxt"])
    weightsPath = os.path.sep.join(['models', "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    model = load_model('models/blood_noblood_classifier.model')

    # input the image
    image = np.array(img)
    orig = image.copy()
    (h, w) = image.shape[:g2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            (blood, noblood) = model.predict(face)[0]
            label = "Blood detected, severly injured" if blood > 0.1 else "NoBlood detected"
            if label == "Blood detected, severly injured":
                bool_label = 1
            else:
                bool_label = 0
    # return the end result
    return jsonify({'Blood detected': bool_label})


if __name__ == '__main__':
    app.run(debug=True)