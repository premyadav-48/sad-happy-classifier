# Sentient Classifier

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
