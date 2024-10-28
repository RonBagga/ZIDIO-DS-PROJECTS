#Internship Projects
This repository contains projects completed during my internship, focusing on deep learning applications in Speech Emotion Recognition and Digit Classification using LSTM and CNN models, respectively. Each project leverages popular datasets and industry-standard libraries to address classification tasks in machine learning.

#Project 1: Speech Emotion Recognition (SER)
##Overview
The Speech Emotion Recognition (SER) project is designed to identify emotions from speech using deep learning techniques. Utilizing the Kaggle RAVDESS dataset, the project employs MFCC (Mel Frequency Cepstral Coefficients) for feature extraction and builds an LSTM (Long Short-Term Memory) model to perform classification across multiple emotion categories.

#Key Components
###Data Source: Kaggle RAVDESS dataset, containing labeled audio files with various emotions.
Feature Extraction: MFCCs, widely used in speech and audio analysis.
Model Architecture: LSTM, chosen for its ability to handle sequential data in time-series problems.
Tools & Libraries: Python, Librosa, Pandas, Seaborn, Matplotlib, TensorFlow, Keras.
Goals
Build a robust pipeline for audio data preprocessing, including noise reduction and feature extraction.
Develop an LSTM model to classify emotions such as neutral, happy, sad, and angry.
Achieve high accuracy in emotion detection from audio input.
Results
The final LSTM model successfully classified emotions with a high degree of accuracy, showcasing the effectiveness of MFCC features in tandem with LSTM for speech-related tasks.

Project 2: Digit Classification
Overview
The Digit Classification project aims to classify handwritten digits using the MNIST dataset. The project employs a Convolutional Neural Network (CNN) to effectively recognize digits from 0 to 9, capitalizing on CNN’s ability to learn spatial hierarchies in image data.

Key Components
Data Source: MNIST dataset, consisting of 28x28 pixel grayscale images of handwritten digits.
Model Architecture: CNN, featuring convolutional and pooling layers to capture essential features in images.
Tools & Libraries: Python, TensorFlow, Keras, Matplotlib.
Goals
Implement a CNN model to automatically recognize and classify digits with high accuracy.
Explore image data preprocessing techniques, including normalization and augmentation.
Fine-tune the CNN model to optimize classification performance on unseen data.
Results
The CNN model achieved high accuracy on the MNIST dataset, demonstrating the CNN's effectiveness for image-based classification tasks.
