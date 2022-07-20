import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
def show_prediction_page():
    st.title("MRI Tumour Detector")
    st.write("""### Upload MRI Image""")
    uploaded_file = st.file_uploader("Choose a file",type=["jpg","png"])
    if uploaded_file is not None:
        img = np.array(Image.open(uploaded_file).convert("RGB").resize((224,224)))/255
        prediction = get_prediction(img)
        st.subheader(f"Tumor Type is {prediction}")
        st.image(img)

@st.cache(suppress_st_warning=True)
def get_prediction(img):
    dec = ["Meningioma","Pituitary","Glioma","No Tumor"]
    model = load_model()
    #pred = np.random.choice(dec)
    pred = model.predict(tf.data.Dataset.from_tensor_slices([img]).batch(1))
    return dec[np.argmax(pred)]
    

@st.cache(allow_output_mutation=True)
def load_model():
    IMG_SHAPE = (224,224,3)
    conv_layer = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                    include_top = False,
                                                    weights = "imagenet")
    conv_layer.trainable = False

    pooling_layer = tf.keras.layers.GlobalAveragePooling2D()
    hidden_layer = tf.keras.layers.Dense(32,activation = "relu")
    output_layer = tf.keras.layers.Dense(4,activation="softmax")
    model = tf.keras.Sequential([
        conv_layer,
        pooling_layer,
        hidden_layer,
        output_layer
    ])
    path = os.path.realpath(__file__)[:-19]
    model.load_weights(f'{path}/checkpoints/mobilenetv2_13epochs')
    return model
    