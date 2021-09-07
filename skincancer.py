import streamlit as st
import numpy as np
import pandas as pd
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D
from keras import backend as K
import time
import io
from PIL import Image
from pathlib import Path
import urllib.request

st.markdown("<h1 style='text-align: center; color: teal;'>Artificial Intelligence Augmented Skin Imaging using Computer Vision and Neural Networks</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: teal;'> By Vinita Silaparasetty</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: teal;'> Msc Data Science at Newcastle University</h1>", unsafe_allow_html=True)

st.sidebar.header("Patient Details")

patient_id = st.sidebar.text_input( "Patient ID:", '#')

st.sidebar.text("Title of Patient:")

choice = st.sidebar.radio("Select the title of the patient:",
                 options=['Mr', 'Mrs', 'Ms', 'Miss', 'Master'])

patient_surname = st.sidebar.text_input("Patient Surname:")

patient_first_name = st.sidebar.text_input("Patient First Name:")

patient_middle_name = st.sidebar.text_input("Patient Middle Name:")

patient_age = st.sidebar.date_input("Select Date of Birth:",value=None)

if st.sidebar.button("Submit"):
	st.sidebar.success("Submission Successful")
else:
	st.sidebar.error("Kindly fill in patient details and click 'Submit'")


def data_gen(x):
    img = np.asarray(Image.open(x).resize((28, 28)))
    x_test = np.asarray(img.tolist())
    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)
    x_test = (x_test - x_test_mean) / x_test_std
    x_validate = x_test.reshape(1, 28, 28, 3)

    return x_validate



def data_gen_(img):
    img = img.reshape(28, 28)
    x_test = np.asarray(img.tolist())
    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)
    x_test = (x_test - x_test_mean) / x_test_std
    x_validate = x_test.reshape(1, 28, 28, 3)

    return x_validate


def load_models():
	with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
		url = 'https://github.com/VinitaSilaparasetty/dissertation/releases/download/maiden/skincancer_98.h5'
		filename = url.split('/')[-1]
		model=urllib.request.urlretrieve(url, filename)
	#model = load_model(filename)
	return model



def predict(x_test, model):
    Y_pred = model.predict(x_test)
    ynew = model.predict_proba(x_test)
    K.clear_session()
    ynew = np.round(ynew, 2)
    ynew = ynew*100
    y_new = ynew[0].tolist()
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    K.clear_session()
    return y_new, Y_pred_classes


st.markdown("<h3 style='text-align: left; color: black;'>Upload Image of Problem Area</h1>", unsafe_allow_html=True)

file_path = st.file_uploader('Upload an image', type=['png', 'jpg','jpeg'])
if file_path is not None:
            x_test = data_gen(file_path)
            image = Image.open(file_path)
            img_array = np.array(image)
            st.header("Image Preview")
            st.success('Upload Successful')
            st.image(img_array,use_column_width=True)
            model = load_models()
            st.header("Diagnosis")
            with st.spinner('Analyzing Image...'):
                    time.sleep(5)
                    y_new, Y_pred_classes = predict(x_test, model)
                    if Y_pred_classes==0:
                             st.success('Patient has Actinic Keratoses')
                    elif Y_pred_classes==1:
                             st.success('Patient has Basal Cell Carcinoma')
                    elif Y_pred_classes==2:
                             st.success('Patient has Benign Keratosis-like Lesions')
                    elif Y_pred_classes==3:
                             st.success('Patient has Dermatofibroma')
                    elif Y_pred_classes==4:
                             st.success('Patient has Melanocytic Nevi')
                    elif Y_pred_classes==5:
                             st.success('Patient has Melanoma')
                    elif Y_pred_classes==6:
                             st.success('Patient has Vascular Lesions')
                    else:
                             st.error("Kindly try another image.")


            
else:
            st.info('Kindly Upload an Image')
                    
