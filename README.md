# TumourDetection
## **Summary**
### Breaking Down the Problem:-
The Problem Statement required us to create a model for detection of Brain Tumour. Brain Tumour is usually detected by a Doctor by observing MRI Scan of the brain,after which several sophisticated medical tests such as biopsy are to confirm the tumour [1]. So we decided to use the MRI scan as input data for our model to predict whether the patient has a brain tumour or not, futhermore, we decided to classify the tumour if we get the data.
### Finding Dataset:-
We searched on Google for a suitable dataset and found a few datasets containing MRI scans of the brain labeled with various types of tumour. First one we found contained 1200 images in total and after scouring around the internet a little bit we found a dataset containing around 7000 images, belonging to 4 different classes which are No Tumour, Meningioma, Glioma and Pituitary, out of which 5700 were for training and 1300 for testing [2].
### Selecting Appropriate tools and Preprocessing Data:-
After we found the data we uploaded it to Google Drive and used it from there by mounting the drive onto a Colab notebook. We used Google Colaboratory since it allows us to use a GPU which makes training faster and allows us to work collectively on the model. We used Pillow module for data preprocessing and then used the Tensorflow module to make our model. Our Data was organized as images put into folders depending upon the class of tumour they belonged to. We made csv files containing information about the image name, image tumour name, and the label encoding of the tumour for both of the training and testing datasets using the Pandas module. We then used the csv files to load the images into RAM and then preprocessed these images by converting them into RGB and into (224.224) shape.
### Learning About Machine Learning, Deep Learning and Transfer Learning:-
We then learned about basic Machine Learning techniques in scikit-learn [3] and slowly moved to learning about Deep Learning and CNNs in Tensorflow [4]. We also came across Transfer Learning Technique which is very popular in Image Classification problems and learning about it through some articles [5] [6].
### Making the Model:-
So now, equipped with knowledge of Machine Learning, Deep Learning and Transfer Learning, We set out to write our model,We used pretrained model to extract features from the images and then added hidden layers for classification. We tried out several different pretrained models for the feature extraction layer such as ResNet50, InceptionV3 , EfficientNetB2 etc and compared their results to select the best for our model. MobileNetV2 turned out to give the best results and gave best results out of all of them.

### Hosting our Model:-
We decided to make a website for showcasing our model. We looked through different options such as flask, Django and some Javascript frameworks but ended up using Streamlit module because it is very easy to use and beginner friendly, moreover it makes deploying the website super easy.

### Additional Models:-
Since we still had time left, we decided to use the same architecture to tackle similar problems such as Covid Detection and Pneumonia Detection using X-Ray scans of chest etc.
## **1.Introduction:**
The given problem statement has been divided into four subparts:

Find a Suitable Dataset for training the Model
Preprocess the data for training.
Make and train the Model.
Make a Website for the Model.
## **2.Dataset Characteristics:**
To create a model to predict brain tumour, we would need some form of data which could be used to identify the tumour. In the medical field, MRI(Magnetic Resonance Imaging) Report is what the doctors use to identify brain tumour, So we decided to look for an Image Dataset which contained labeled images of MRI Scans of the Brain. We found a Dataset on Kaggle,it had 5700 train images and 1300 test images,almost evenly distributed across the following types:
i) No Tumor
ii) Pituitary
iii) Glioma
iv) Meningioma
The images were all mri images which are grayscale.The size of the images was (512X512).

### 2.1 Data Preprocessing
We uploaded the data into a google drive,then accessed the data in google collab,we created a csv file for easier access to the data and for label encoding,then using the csv file we accessed  the images and resized them into the proper image shape of (224,224,3).We use this shape as it is the best suited image shape we can use for the pretrained cnn models we would use for our model.

## **Model**
 Inception  V3
The Inception V3 is a deep learning model for image classification that is built on convolutional neural networks. The Inception V3 is an improved version of the Inception V1 basic model, which was presented as GoogleNet in 2014 [1]. Inception V3's model architecture can be seen in Figure 1.
