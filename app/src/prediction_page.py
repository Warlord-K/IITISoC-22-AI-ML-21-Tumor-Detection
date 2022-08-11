import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import time

def init():
    st.session_state.tumour_model = load_model()
    st.session_state.selected = False

def main():
    if "tumour_model" not in st.session_state:
        init()
    st.title("MRI Tumour Detector")
    st.write("""### Upload MRI Image""")
    uploaded_file = st.file_uploader("",type=["jpg","png"])
    if uploaded_file is not None:
        img = np.array(Image.open(uploaded_file).convert("RGB").resize((224,224)))/255
        get_prediction(img)
        #st.subheader(f"Tumor Type is {prediction}")
    if st.session_state.selected:
        prog_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            prog_bar.progress(i+1)
        prog_bar.empty()
        st.subheader(f"Prediction is {st.session_state.prediction}")
        if st.session_state.prediction == "Meningioma":
            st.error('The Model indicates that the patient has Meningioma tumour, Meningioma is a tumor that forms on membranes that covers the brain and spinal cord just inside the skull.Overall, meningiomas are the most common type of primary brain tumor. ') 
            st.markdown('##### <a href = "https://www.cancer.gov/rare-brain-spine-tumor/tumors/meningioma#:~:text=A%20meningioma%20is%20a%20primary,grade%20meningiomas%20are%20very%20rare."> Know more🔗</a>',unsafe_allow_html=True)  
        if st.session_state.prediction == "Pituitary":
            st.error('The Model indicates that the patient has Pituitary tumour, Pituitary tumours are abnormal growths that develop in your pituitary gland.')
            st.markdown('##### <a href = "https://www.mayoclinic.org/diseases-conditions/pituitary-tumors/symptoms-causes/syc-20350548#:~:text=Pituitary%20tumors%20are%20abnormal%20growths,produce%20lower%20levels%20of%20hormones."> Know more🔗</a>',unsafe_allow_html=True)  
        if st.session_state.prediction == "Glioma":
             st.error('The Model indicates that the patient has Glioma tumour, Glioma is a type of tumor that occurs in the brain and spinal cord.') 
             st.markdown('##### <a href = "https://www.mayoclinic.org/diseases-conditions/glioma/symptoms-causes/syc-20350251#:~:text=Glioma%20is%20a%20type%20of,glial%20cells%20can%20produce%20tumors."> Know more🔗</a>',unsafe_allow_html=True)  
        if st.session_state.prediction == "No Tumor":
            st.success("Congratulations! You have No Tumour.")
        st.image(st.session_state.img)

        
    with st.expander("Don't have any MRI Scans?"):
        path = os.path.realpath(__file__)[:-22]
        ims = np.random.choice(os.listdir(f"{path}/test_files"),3,replace = False)
        labels = []
        images = [np.array(Image.open(f"{path}/test_files/{i}").convert("RGB").resize((224,224)))/255 for i in ims]
        imgs = st.columns([1,1,1])
        for i,img in enumerate(imgs):
            if ims[i][3:5] == 'no':
                labels.append("No Tumour")
            elif ims[i][3:5] == 'pi':
                labels.append("Pituitary")
            elif ims[i][3:5] == 'gl':
                labels.append("Glioma")
            else:
                labels.append("Meningioma")

            img.image(images[i])
            img.button(f" ({i+1}){labels[i]}",on_click = get_prediction,args =(images[i],))
            
      
        

    


def get_prediction(img):
    dec = ["Meningioma","Pituitary","Glioma","No Tumor"]
    #pred = np.random.choice(dec)
    
    pred = st.session_state.tumour_model.predict(tf.data.Dataset.from_tensor_slices([img]).batch(1))
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
    path = os.path.realpath(__file__)[:-22]
    model.load_weights(f'{path}/checkpoints/mobilenetv2_13epochs')
    return model


if __name__ == "__main__":
    main()
    