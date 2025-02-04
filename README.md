# Artificial Intelligence Augmented Skin Imaging using Computer Vision and Convolutional Neural Networks
Prototype for software application to promote contactless diagnosis of skin cancer.

### Aim:

To use artificial intelligence to develop a smart skin imaging application that understands the skin’s reaction to the sun by categorizing it into it’s Fitzpatrick phototype and subsequently for the automatic, contactless diagnosis of skin cancer using low resolution images of skin lesions taken with a mobile device.

### Objectives:

* To develop a software application that automates the process of skin cancer diagnosis and eliminates the need for high resolution images of the problem area by training a convolutional neural network to achieve accurate results with sensitivity and specificity of values of 1.

* To develop a skin imaging software application for the diagnosis of cancerous skin lesions with a validation accuracy of at least 85\% .

* To develop a skin imaging software application that will efficiently categorize the Fitzpatrick skin phototype of individuals from all ethnic groups. Understanding the phototype of the patient is helpful, but it is not as life threatening if the classification is wrong. Hence for a prototype, 75\% accuracy is ideal.

### Additional Details
* Used Python's imblearn package for random under-sampling to balance the dataset. 
* Used Python's Matplotlib and Seaboard to generate visuals for better data understanding. 
* Used Python's Scikit - Learn to perform K-means to detect and cluster the colored pixels in an image. The elbow method was used to select the ideal number of clusters. The RGB values of each unique color within the image were converted to hex values. Finally,the Fitzpatrick type was determined using the hex values for skin tones in each Fitzpatrick type. 

### Instructions to run prototype: 

* Ensure that the file 'skincancer.py' and the keras model file 'skincancer_98.h5' are in the same working directory as your streamlit installation.

* In the terminal type the command 'streamlit run skincancer.py'

* A new tab will open in your browser with the application gui.
