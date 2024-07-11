import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import preprocess_input

from flask import Flask , request, render_template

from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model("covidVision_model.h5")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/test')
def test():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/precautions')
def predictions():
    return render_template("precautions.html")

@app.route('/trends')
def trends():
    return render_template("trends.html")

@app.route('/predict',methods=["POST"])
def predict():
    file = request.files['image']
    filename = secure_filename(file.filename)
    basepath=os.path.dirname(__file__)
    print('current path : ', basepath)

    filepath= os.path.join(basepath,'static/uploads',filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print('file path : ', filepath)
    file.save(filepath)

    img = load_img(filepath, target_size=(299,299))
    x=img_to_array(img)
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x)

    pred=np.argmax(model.predict(x),axis=1)
    index=['Covid-19', 'Lung Opacity', 'Normal', 'Viral Pneumonia']
    result=index[pred[0]]
    text = "The person is diagnosised with : " + str(result)
    return render_template("index.html",result=text)

if __name__=="__main__":
    app.run(debug=False, threaded=False)
    