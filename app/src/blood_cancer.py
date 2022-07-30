import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

def init():
    st.session_state.cell_model = load_model()
    st.session_state.selected = False

def main():
    if "cell_model" not in st.session_state:
        init()
    st.title("Blood Cell Detector")
    st.write("""### Upload Blood Cell Image""")
    uploaded_file = st.file_uploader("",type=["jpeg","png","jpg"])
    if uploaded_file is not None:
        img = np.array(Image.open(uploaded_file).convert("RGB").resize((224,224)))/255
        prediction = get_prediction(img)
        #st.subheader(f"Tumor Type is {prediction}")
    if st.session_state.selected:
        st.subheader(f"Prediction is {st.session_state.prediction}")
        st.image(st.session_state.img)

        
    with st.expander("Don't have any Blood Cell Images?"):
        path = os.path.realpath(__file__)[:-20]
        ims = np.random.choice(os.listdir(f"{path}/testfinal"),3,replace = False)
        labels = []
        images = [np.array(Image.open(f"{path}/testfinal/{i}").convert("RGB").resize((224,224)))/255 for i in ims]
        imgs = st.columns([1,1,1])
        for i,img in enumerate(imgs):
            if ims[i][0:3] == 'eos':
                labels.append("Eosinophil")
            elif ims[i][0:3] == 'lym':
                labels.append("Lymphocyte")
            elif ims[i][0:3] == 'mon':
                labels.append("Monocyte")
            else:
                labels.append("Nuetrophil")

            img.image(images[i])
            img.button(f" ({i+1}){labels[i]}",on_click = get_prediction,args =(images[i],))
        

    


def get_prediction(img):
    dec = ["Lymphocyte","Neutrophil","Eosinophil","Monocyte"]
    #pred = np.random.choice(dec)
    pred = st.session_state.cell_model.predict(tf.data.Dataset.from_tensor_slices([img]).batch(1))
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
    output_layer = tf.keras.layers.Dense(4,activation="softmax")
    model = tf.keras.Sequential([
        conv_layer,
        pooling_layer,
        hidden_layer,
        output_layer
    ])
    path = os.path.realpath(__file__)[:-20]
    model.load_weights(f'{path}/checkpoints/bloodcell_24epochs')
    return model


if __name__ == "__main__":
    main()
    