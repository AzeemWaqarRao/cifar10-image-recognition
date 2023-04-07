# Cifar10 Image Classifier using Transfer Learning
This is a convolutional neural network (CNN) project in Python that classifies images from the Cifar10 dataset. The model used is ResNet-50, which has been trained on the dataset and saved as "resnet.h5" file. A Streamlit app has been created to load the model and predict the class of an image uploaded by the user. [Click here to visit app](https://azeemwaqarrao-cifar10-image-recognition-streamlit-app-v8eci1.streamlit.app/)

## Dataset
The Cifar10 dataset consists of 50,000 32x32 color images in 10 classes, with 5,000 images per class. The classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. In this project, the images in dataset used have been resized to 124x124 for better accuracy.

## Transfer Learning
Transfer Learning is a technique where a pretrained model is used for a project. Resnet-50 is a model trained on imagenet dataset (which contains over a Million images belonging to 1000 classes). In this project resnet-50 is redesigned to have 10 possible outputs corresponding to 10 classes and trained again on the given data.

## Dependencies
To run the Streamlit app, you need to have the following dependencies installed:

Python 3.6+<br>
tensorflow==2.4.1<br>
keras==2.4.3<br>
streamlit==0.80.0<br>
Pillow==8.2.0<br>

You can install them by running:

```
pip install -r requirements.txt
```

## Usage
To run the app, use the following command:

```
streamlit run app.py
```

This will open the app in your web browser. You can then upload an image and click the "Predict" button to see the predicted class.

## Files
* **cifar_10_model_training.ipynb:** Jupyter notebook used to train the ResNet-50 model on the Cifar10 dataset.
* **resnet.h5**: Saved model file.
* **streamlit_app.py**: Streamlit app code to load the model and predict the class of an image.
* **requirements.txt**: List of Python dependencies required for the app.

## Credits
This project was completed as part of the Kaggle Cifar10 dataset competition. 




