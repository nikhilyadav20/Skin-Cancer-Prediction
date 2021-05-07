from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

from tensorflow.keras.models import Model , load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

Model= load_model('skin_model.h5')

lesion_classes_dict = {
    0: 'Actinic keratoses',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic nevi',
    6: 'Vascular lesions'
}

app = Flask(__name__)

def model_predict(img_path, Model):
    img = image.load_img(img_path, target_size=(64,64,3))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    preds = Model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path , Model)

        # Process your result for human
        

        pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   
        pr = lesion_classes_dict[pred_class[0]]
        result =str(pr)         
        return result
    return None

if __name__ == "__main__":
    app.run(debug=True)