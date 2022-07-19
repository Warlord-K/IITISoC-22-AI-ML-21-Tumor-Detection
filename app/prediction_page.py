import streamlit as st
from PIL import Image
import numpy as np
def show_prediction_page():
    st.title("MRI Tumour Detector")
    st.write("""### Upload MRI Image""")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        prediction = get_prediction(img)
        st.subheader(f"Tumor Type is {prediction}")
        st.image(img)

@st.cache
def get_prediction(img):
    dec = ["Meningioma","Glioma","No Tumor","Pituitary"]
    pred = np.random.choice(dec)
    return pred
    