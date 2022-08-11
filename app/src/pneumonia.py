import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import time
def init():
    st.session_state.pneumonia_model = load_model()
    st.session_state.selected = False

def main():
    if "pneumonia_model" not in st.session_state:
        init()
    st.title("Pneumonia Detector")
    st.write("""### Upload Chest Xray Image""")
    
    uploaded_file = st.file_uploader("",type=["jpg","png"])
    if uploaded_file is not None:
        img = np.array(Image.open(uploaded_file).convert("RGB").resize((224,224)))/255
        prediction = get_prediction(img)
        #st.subheader(f"Tumor Type is {prediction}")

        

    if st.session_state.selected:
        prog_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            prog_bar.progress(i+1)
        prog_bar.empty()
        st.subheader(f"Prediction is {st.session_state.prediction}")
        if st.session_state.prediction == "Bacteria Pneumonia":
            st.error('The model indicates that the patient has Bacterial Pneumonia,Bacterial pneumonia is an infection of your lungs caused by certain bacteria. The most common one is Streptococcus (pneumococcus), but other bacteria can cause it too.')
            st.markdown('##### <a href = "https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/syc-20354204#:~:text=and%20your%20lungs-,Pneumonia%20and%20your%20lungs,within%20your%20lungs%20(alveoli).">follow this link to know more.</a>',unsafe_allow_html=True)
        if st.session_state.prediction == "Virus Pneumonia":
            st.error('The model indicates that the patient has Virus Pneumonia,Viral pneumonia is an infection of your lungs caused by a virus. The most common cause is the flu, but you can also get viral pneumonia from the common cold and other viruses.')
            st.markdown('##### <a href = "https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/syc-20354204#:~:text=and%20your%20lungs-,Pneumonia%20and%20your%20lungs,within%20your%20lungs%20(alveoli).">follow this link to know more.</a>',unsafe_allow_html=True)        
        if st.session_state.prediction == "Normal":
            st.success('Congratulations,the model indicates that the patient is normal,if you seem to get any symptoms please consult a doctor.')
        st.image(st.session_state.img)

        
    with st.expander("Don't have any XRay Scans?"):
        path = os.path.realpath(__file__)[:-16]
        ims = np.random.choice(os.listdir(f"{path}/test_files_chest_xray"),3,replace = False)
        labels = []
        images = [np.array(Image.open(f"{path}/test_files_chest_xray/{i}").convert("RGB").resize((224,224)))/255 for i in ims]
        imgs = st.columns([1,1,1])
        for i,img in enumerate(imgs):
            if ims[i][0] == 'b':
                labels.append("Pneumonia Bacteria")
            elif ims[i][0] == 'n':
                labels.append("Normal")
            elif ims[i][0] == 'v':
                labels.append("Virus")
            # else:
            #     labels.append("Meningioma")

            img.image(images[i])
            img.button(f" ({i+1}){labels[i]}",on_click = get_prediction,args =(images[i],))
        

    


def get_prediction(img):
    dec = ["Virus Pneumonia","Bacteria Pneumonia","Normal"]
    #pred = np.random.choice(dec)
    pred = st.session_state.pneumonia_model.predict(tf.data.Dataset.from_tensor_slices([img]).batch(1))
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
    path = os.path.realpath(__file__)[:-16]
    model.load_weights(f'{path}/checkpoints/pnemonia_final_epochs')
    return model


if __name__ == "__main__":
    main()
    