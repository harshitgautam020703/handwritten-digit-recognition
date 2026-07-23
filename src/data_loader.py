import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        
    def load_data(self):
        """Load and preprocess MNIST dataset"""
        print("Loading MNIST dataset...")
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        
        # Normalize pixel values to [0,1]
        self.X_train = self.X_train.astype('float32') / 255.0
        self.X_test = self.X_test.astype('float32') / 255.0
        
        # Reshape for CNN (add channel dimension)
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 28, 28, 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 28, 28, 1)
        
        # One-hot encoding for labels
        self.y_train = tf.keras.utils.to_categorical(self.y_train, 10)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, 10)
        
        # Split training data into train and validation (80-20)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Validation set: {self.X_val.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
    
    def visualize_samples(self, num_samples=10):
        """Visualize sample digits"""
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(num_samples):
            idx = np.random.randint(0, len(self.X_train))
            axes[i].imshow(self.X_train[idx].reshape(28, 28), cmap='gray')
            axes[i].set_title(f'Label: {np.argmax(self.y_train[idx])}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('outputs/sample_digits.png')
        plt.show()