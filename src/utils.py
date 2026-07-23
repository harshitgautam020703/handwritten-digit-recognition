import os
import json
import pickle
import numpy as np
import tensorflow as tf

class Utils:
    @staticmethod
    def save_model(model, path='models/mnist_model.h5'):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.save(path)
        print(f"✅ Model saved to {path}")
    
    @staticmethod
    def load_model(path='models/mnist_model.h5'):
        """Load model from disk"""
        return tf.keras.models.load_model(path)
    
    @staticmethod
    def save_history(history, path='outputs/history.pkl'):
        """Save training history"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(history.history, f)
        print(f"✅ History saved to {path}")
    
    @staticmethod
    def load_history(path='outputs/history.pkl'):
        """Load training history"""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_config(config, path='config/config.json'):
        """Save configuration"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Config saved to {path}")
    
    @staticmethod
    def set_seed(seed=42):
        """Set random seeds for reproducibility"""
        np.random.seed(seed)
        tf.random.set_seed(seed)