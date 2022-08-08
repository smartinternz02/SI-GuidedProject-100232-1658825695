#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests 
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.models import load_model
app = Flask(__name__, template_folder='template')
model_fruit = load_model("fruit.h5")
model_veg = load_model("Vegetable.h5")


app = Flask(__name__)
@app.route('/')
def home ():
    return render_template('home.html')

@app.route('/prediction')
def prediction():
    return render_template('predict.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        print("File_path", f)
        img = image.load_img(file_path,target_size=(128,128))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        plant=request.form['plant']
        print(plant)
        if (plant=="vegetable"):
            preds = model_veg.predict(x)
            Prediction = np.argmax(preds)
            df=pd.read_excel('precautionveg.xlsx')
            print(df.iloc[Prediction]['caution'])
            str_1 = df.iloc[Prediction]['caution']  
        else:
            preds = model_fruit.predict(x)
            Prediction = np.argmax(preds)
            df=pd.read_excel('precautionfruits.xlsx')
            print(df.iloc[Prediction]['caution'])
            str_1 = df.iloc[Prediction]['caution']            
        return str(str_1)

    
if __name__=="__main__":
    app.run(debug=False)