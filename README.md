# MaskDetectionProject
# Dataset
You can access images used for training-validation and test from here:https://drive.google.com/drive/folders/1zF3IsClw-Uh6B303v-AwPe3p250ghSfF?usp=sharing<br/>
Please note that since you download the folders form link above, you have to store them into a folder named dataset on the same folder with you project
# Training
In case you want to train the model by yourself just run CNNClassifier_ImageDataGenerator.py. This file except from training, splits and the data into corresponding folders 
# Explore the app
In case you want to run our application you have to follow the next steps, after downloading this repository: <br/>
* Video Application: contains files needed for web application
  * app.py: launces the application
  * base_camera.py: Class needed to launch camera on web browser
  * camera.py: python file that uses our trained model to make predictions and demonstrate them on camera
  * log.txt: log file for the camera
  * templates: contains html code for web application
  * static: contains javascript and css files for web application
* CNNClassifier_ImageDataGenerator.py: file used to train our basic model
* SVM classifier.py: file used to train an SVM classifier
* cnn_model.h5: basic model used for prediction
* haarcascade_frontalface_default.xml: xml used for face detection
* Mask_Project_with_cropped_images.ipynb: Ipython notebook that is used to train our third try-model for image classification.
* ImageTransformation.ipynb: This file is used to cropp the images that will be fed into the third model.
* MaskDetectionProjectReport.pdf: Final report of our project.
# Video Demonstration
In case you want to run our application and see a live demonstration you have to run the app.py file inside VideoApplication folder.<br/>
After running this file open your browser on http://localhost:5000 and our application will launch on your browser. This application uses cnn_model.h5 file which contains our model. Bellow you can see a live demonstration
<br/>
<br/>
![](mask_web_app.gif)
