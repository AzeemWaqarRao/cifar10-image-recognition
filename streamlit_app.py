import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from tensorflow.keras.utils import img_to_array


model =load_model('inception.h5')
cifar10_labels = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


def prediction(image):
    image = image.resize((128,128))
    image = img_to_array(image)/255
    image = np.expand_dims(image, axis=0)
    pred = np.argmax(model.predict(image))
    return "This Image is of a " + cifar10_labels[pred]


def main():
    st.title("Cifar 10 Image Recognition")
    img_file = st.file_uploader("Upload Image")

    if img_file is not None:
        image = Image.open(img_file)
        result = prediction(image)
        st.success(result)
        st.image(image, caption='uploaded image', use_column_width=True)


if __name__ == "__main__":
    main()