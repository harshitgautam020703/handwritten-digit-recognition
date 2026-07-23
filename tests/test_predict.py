import numpy as np
from src.data_loader import DataLoader
from src.model import DigitRecognizerModel
from src.predict import Predictor

def test_prediction():
    """Test prediction functionality"""
    # Load data and train model quickly
    data_loader = DataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.load_data()
    
    model_builder = DigitRecognizerModel()
    model = model_builder.build_model()
    model = model_builder.compile_model(lr=0.001)
    
    # Train on small subset
    model.fit(X_train[:500], y_train[:500], epochs=3, verbose=0)
    
    predictor = Predictor(model)
    
    # Test single prediction
    pred, conf = predictor.predict_image(X_test[0].reshape(28, 28, 1))
    assert 0 <= pred <= 9
    assert 0 <= conf <= 100
    print(f"✅ Single prediction test passed: {pred}, {conf:.2f}%")
    
    # Test batch prediction
    predictor.predict_batch(X_test[:5])
    print("✅ Batch prediction test passed")

if __name__ == "__main__":
    test_prediction()
    print("\n✅ All prediction tests passed!")