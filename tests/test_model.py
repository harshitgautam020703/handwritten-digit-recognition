import tensorflow as tf
import numpy as np
from src.data_loader import DataLoader
from src.model import DigitRecognizerModel

def test_model_architecture():
    """Test model building"""
    model = DigitRecognizerModel()
    model.build_model()
    assert model.model is not None
    assert len(model.model.layers) > 0
    print("✅ Model architecture test passed")

def test_model_training():
    """Test model training on small subset"""
    data_loader = DataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.load_data()
    
    # Use small subset
    X_subset = X_train[:100]
    y_subset = y_train[:100]
    
    model_builder = DigitRecognizerModel()
    model = model_builder.build_model()
    model = model_builder.compile_model(lr=0.001)
    
    history = model.fit(X_subset, y_subset, epochs=2, verbose=0)
    assert len(history.history['loss']) == 2
    print("✅ Model training test passed")

def test_model_save_load():
    """Test model saving and loading"""
    model = DigitRecognizerModel()
    model.build_model()
    model.compile_model()
    
    # Save and load
    model.model.save('test_model.h5')
    loaded = tf.keras.models.load_model('test_model.h5')
    
    assert loaded is not None
    print("✅ Model save/load test passed")
    
    # Cleanup
    import os
    os.remove('test_model.h5')

if __name__ == "__main__":
    test_model_architecture()
    test_model_training()
    test_model_save_load()
    print("\n✅ All tests passed!")