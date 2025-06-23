# Sentient Classifier: Sad vs Happy

A deep learning image classifier that predicts whether a person is **sad** or **happy** using TensorFlow and convolutional neural networks (CNN).

---

## Project Overview

This project builds a binary image classifier to distinguish between sad and happy facial expressions. The model is trained on labeled images and can predict the emotional state from new photos.

---

## Features

- Data preprocessing and cleaning to remove corrupted images  
- Image dataset loading with TensorFlow's `image_dataset_from_directory`  
- Image scaling and train/validation/test split  
- CNN architecture built with Keras Sequential API  
- Model training with TensorBoard callback  
- Performance evaluation with precision, recall, and accuracy metrics  
- Model saving and loading for future predictions  
- Sample test prediction on a single image  

---

## Requirements

- Python 3.11  
- TensorFlow 2.19.0  
- OpenCV  
- Pillow  
- Matplotlib  
- Other dependencies listed in `requirements.txt`

To install dependencies, run:

```bash
pip install -r requirements.txt


Usage
Clone the repo:

bash
Copy
Edit
git clone git@github.com:premyadav-48/sad-happy-classifier.git
cd sad-happy-classifier
Prepare your data inside the data directory with two subfolders: happy and sad.

Run the notebook or script to train the model:

bash
Copy
Edit
jupyter notebook SadHappy_Classifier.ipynb
Use the trained model to predict emotions from new images.

Project Structure
bash
Copy
Edit
sad-happy-classifier/
│
├── data/                 # Dataset folders (happy/ and sad/)
├── models/               # Saved models
├── logs/                 # TensorBoard logs
├── SadHappy_Classifier.ipynb  # Main notebook with training and evaluation
├── requirements.txt      # Python dependencies
├── .gitignore            # Git ignore file
└── README.md             # This file
How It Works
Data Cleaning: Removes invalid or corrupt images.

Loading Data: Uses TensorFlow's image_dataset_from_directory for easy pipeline creation.

Preprocessing: Scales images to [0, 1] range and splits dataset.

Model: CNN with 3 convolutional layers, max pooling, dense layers, and sigmoid output.

Training: Uses binary cross-entropy loss and Adam optimizer, trained for 20 epochs.

Evaluation: Reports precision, recall, and accuracy on test data.

Inference: Resizes input images and predicts emotional state.

Example Prediction
python
Copy
Edit
import cv2
import tensorflow as tf
import numpy as np

img = cv2.imread('sadtest.jpg')
resize = tf.image.resize(img, (256,256))
yhat = model.predict(np.expand_dims(resize/255, 0))

if yhat > 0.5:
    print('Predicted class is Sad')
else:
    print('Predicted class is Happy')
