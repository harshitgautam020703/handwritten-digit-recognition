import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class Trainer:
    def __init__(self, model, data_loader, batch_size=128, epochs=50):
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.epochs = epochs
        self.history = None
        
    def train(self):
        """Train the model"""
        print("Starting training...")
        start_time = datetime.now()
        
        self.history = self.model.fit(
            self.data_loader.X_train,
            self.data_loader.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.data_loader.X_val, self.data_loader.y_val),
            callbacks=self.model.get_callbacks(),
            verbose=1
        )
        
        training_time = datetime.now() - start_time
        print(f"Training completed in {training_time}")
        
        return self.history
    
    def evaluate(self):
        """Evaluate model on test set"""
        test_loss, test_accuracy = self.model.evaluate(
            self.data_loader.X_test,
            self.data_loader.y_test,
            verbose=0
        )
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        return test_loss, test_accuracy
    
    def plot_training_history(self):
        """Plot training and validation accuracy/loss"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('outputs/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def confusion_matrix_plot(self):
        """Plot confusion matrix"""
        y_pred = self.model.predict(self.data_loader.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.data_loader.y_test, axis=1)
        
        cm = confusion_matrix(y_true, y_pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes))