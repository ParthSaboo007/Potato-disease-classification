import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
import os
import tensorflow as tf
from keras.preprocessing import image

st.title("welcome TO Potato disese classification model")

MODEL = tf.keras.models.load_model('E:/DS/PROJECTS/potato_disease_classification/streamlit_webapp/2')
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        image = read_file_as_image(image_data)
        img_batch = np.expand_dims(image, 0)
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence_score = np.max(predictions[0])
        st.write(predicted_class)
        st.write('confidence = ',confidence_score*100,'%')
        st.image(image_data,caption = predicted_class)


def main():
    st.title('Upload the potato leaf image to be analysed')
    load_image()


if __name__ == '__main__':
    main()
#st.title(model1.summary())