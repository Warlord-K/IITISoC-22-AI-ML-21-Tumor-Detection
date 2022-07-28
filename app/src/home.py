import streamlit as st


def main():
    st.markdown(
        '''
        <h1 align="center">
            Disease Detection Using Convolutional Neural Networks
        </h1>

        ---

        #### Project Solution
        
        <h2>
    
        -	We will take the image dataset and annotations and create a csv file containing filename and type of tumor.
        -	We will pass the image to a Convolutional Deep Neural Network. The Model will be comprised of Conv2D layers, a Pooling layer and Few dense layers at the end for classification. We will try out a bunch of combinations of convolutional layers to get the best results. We will use Adam as our optimizer.
        -	We will use RELU for activations and Categorical Cross Entropy as the Loss function. We will take specificity and sensitivity as our metrics.
        -	We will also try out some pretrained models for the convolutional layer such as MobileNetV2, ResNet50, InceptionV3 etc. and compare their results with our custom model.
        -	If we get the data for it, we will also try to create a model which makes a bounding box around the region which has the tumor using YOLO architecture.

        </h2>

        ---

        #### Project Timeline

        <h2>
        
        -	In the first week, we will find and process the dataset. We will create a csv file containing 2 columns, one for the image path and one for the type of tumor,we will simply write “no” if there is no tumor.
        -	In the second week, we will make our custom deep learning model using TensorFlow Keras.
        -	In the third week, we will compare the results of our custom convolutional model with pretrained models like MobileNet, ResNet and Inception etc.
        -	In the fourth week,we will wrap up the project and if time permits we will create a website on which MRI image could be uploaded and a prediction could be seen.

        </h2>
        
        ''',
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()
