# Handwritten Digit Recognition using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-97.13%25-brightgreen.svg)]()

> A production-ready deep learning pipeline for handwritten digit recognition achieving 97.13% validation accuracy on the MNIST dataset.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Deployment](#deployment)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits (0-9) using the MNIST dataset. The model is designed with a focus on:

- **High Accuracy**: 97.13% validation accuracy
- **Efficient Training**: Optimized architecture with batch normalization and dropout
- **Production Ready**: Clean code structure with deployment options
- **Comprehensive Evaluation**: Detailed metrics and visualizations

## ✨ Features

- ✅ **Data Pipeline**: Automatic loading, normalization, and one-hot encoding
- ✅ **CNN Architecture**: 3 convolutional blocks with batch normalization
- ✅ **Regularization**: L2 regularization, dropout, and early stopping
- ✅ **Optimization**: Adam optimizer with learning rate scheduling
- ✅ **Visualization**: Training history, confusion matrix, and sample predictions
- ✅ **Deployment Ready**: Flask and FastAPI implementations
- ✅ **Testing**: Comprehensive unit tests
- ✅ **Reproducibility**: Fixed random seeds for consistent results

## 📁 Project Structure

handwritten-digit-recognition/
├── src/ # Source code
│ ├── init.py # Package initialization
│ ├── data_loader.py # Data loading and preprocessing
│ ├── model.py # Model architecture
│ ├── train.py # Training pipeline
│ ├── predict.py # Prediction module
│ └── utils.py # Utility functions
├── tests/ # Unit tests
│ ├── test_model.py # Model tests
│ └── test_predict.py # Prediction tests
├── deployment/ # Deployment options
│ ├── app.py # Flask web app
│ └── api.py # FastAPI implementation
├── config/ # Configuration files
│ └── config.yaml # Model configuration
├── models/ # Saved models
│ └── mnist_model.h5 # Trained model
├── outputs/ # Generated outputs
│ ├── training_history.png # Training plots
│ ├── confusion_matrix.png # Confusion matrix
│ └── sample_digits.png # Sample visualizations
├── notebooks/ # Jupyter notebooks (optional)
│ └── 01_data_exploration.ipynb
├── main.py # Main execution script
├── requirements.txt # Python dependencies
├── setup.py # Package setup
├── .gitignore # Git ignore file
└── README.md # This file
