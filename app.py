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

st.markdown("<h1 style='text-align: center; color: teal;'>Artificial Intelligence Augmented Skin Imaging using Computer Vision and Neural Networks</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: teal;'> By Vinita Silaparasetty</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: teal;'> Msc Data Science </h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: teal;'> Newcastle University </h1>", unsafe_allow_html=True)

st.write("This is a prototype for an application to aid in the contactless diagnoses of skin cancer.")

model_path = './models/'

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

    model = load_model(model_path + 'skincancer_98.h5')
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


st.header("Upload Image")
file_path = st.file_uploader('Upload an image', type=['png', 'jpg'])
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
                    time.sleep(2)
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
                             st.info("Kindly try another image.")


            
else:
            st.info('Kindly Upload an Image')
                    
