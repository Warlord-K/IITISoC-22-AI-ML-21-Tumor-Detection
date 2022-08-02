import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

def init():
    st.session_state.covid_model = load_model()
    st.session_state.selected = False

def main():
    if "covid_model" not in st.session_state:
        init()
    st.title("covid19 Detector")
    st.write("""### Upload MRI Image""")
    uploaded_file = st.file_uploader("",type=["jpg","png"])
    if uploaded_file is not None:
        img = np.array(Image.open(uploaded_file).convert("RGB").resize((224,224)))/255
        prediction = get_prediction(img)
        #st.subheader(f"Tumor Type is {prediction}")
    if st.session_state.selected:
        st.subheader(f"Prediction is {st.session_state.prediction}")
        st.image(st.session_state.img)

        
    with st.expander("Don't have any MRI Scans?"):
        path = os.path.realpath(__file__)[:-14]
        ims = np.random.choice(os.listdir(f"{path}/testcovid"),3,replace = False)
        labels = []
        images = [np.array(Image.open(f"{path}/testcovid/{i}").convert("RGB").resize((224,224)))/255 for i in ims]
        imgs = st.columns([1,1,1])
        for i,img in enumerate(imgs):
            if ims[i][0:2] == 'CO':
                labels.append("covid")
            elif ims[i][0:2] == 'No':
                labels.append("Normal")
            elif ims[i][0:2] == 'Vi':
                labels.append("pneumonia")
        
            img.image(images[i])
            img.button(f" ({i+1}){labels[i]}",on_click = get_prediction,args =(images[i],))
        

    


def get_prediction(img):
    dec = ["covid","normal","viral pneumonia"]
    #pred = np.random.choice(dec)
    pred = st.session_state.covid_model.predict(tf.data.Dataset.from_tensor_slices([img]).batch(1))
    prediction = dec[np.argmax(pred)]
    st.session_state.prediction = prediction
    st.session_state.img = img
    st.session_state.selected = True
    

def load_model():
    IMG_SHAPE = (224,224,3)
    conv_layer = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                    include_top = False,
                                                    weights = "imagenet")
    conv_layer.trainable = False

    pooling_layer = tf.keras.layers.GlobalAveragePooling2D()
    hidden_layer = tf.keras.layers.Dense(32,activation = "relu")
    output_layer = tf.keras.layers.Dense(3,activation="softmax")
    model = tf.keras.Sequential([
        conv_layer,
        pooling_layer,
        hidden_layer,
        output_layer
    ])
    path = os.path.realpath(__file__)[:-14]
    model.load_weights(f'{path}/checkpoints/50epoch0001')
    
    return model


if __name__ == "__main__":
    main()
    