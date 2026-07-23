import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

class Predictor:
    def __init__(self, model):
        self.model = model
        
    def predict_single(self, image_path):
        """Predict digit from single image"""
        # Load and preprocess image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))
        img_array = np.array(img)
        
        # Invert if necessary (MNIST has white digits on black background)
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        
        # Normalize and reshape
        img_array = img_array.astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Predict
        prediction = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        # Display
        plt.figure(figsize=(4, 4))
        plt.imshow(img_array.reshape(28, 28), cmap='gray')
        plt.title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2f}%')
        plt.axis('off')
        plt.show()
        
        return predicted_class, confidence
    
    def predict_batch(self, images, num_samples=5):
        """Predict and visualize multiple images"""
        indices = np.random.choice(len(images), num_samples, replace=False)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        
        for i, idx in enumerate(indices):
            img = images[idx]
            prediction = self.model.predict(img.reshape(1, 28, 28, 1), verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            axes[i].imshow(img.reshape(28, 28), cmap='gray')
            axes[i].set_title(f'Pred: {predicted_class}\nConf: {confidence:.1f}%')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('outputs/batch_predictions.png')
        plt.show()