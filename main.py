import os
import tensorflow as tf
from src.data_loader import DataLoader
from src.model import DigitRecognizerModel
from src.train import Trainer
from src.predict import Predictor
from src.utils import Utils

def main():
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Set seed
    Utils.set_seed(42)
    
    # 1. Load Data
    print("="*50)
    print("1. Loading Data")
    print("="*50)
    data_loader = DataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.load_data()
    data_loader.visualize_samples(save_path='outputs/sample_digits.png')
    
    # 2. Build Model
    print("\n" + "="*50)
    print("2. Building Model")
    print("="*50)
    model_builder = DigitRecognizerModel()
    model = model_builder.build_model()
    model = model_builder.compile_model(lr=0.001)
    model_builder.summary()
    
    # 3. Train Model
    print("\n" + "="*50)
    print("3. Training Model")
    print("="*50)
    trainer = Trainer(model, data_loader, batch_size=128, epochs=50)
    history = trainer.train()
    
    # 4. Evaluate
    print("\n" + "="*50)
    print("4. Evaluating Model")
    print("="*50)
    test_loss, test_acc = trainer.evaluate()
    
    # 5. Visualize
    print("\n" + "="*50)
    print("5. Visualizing Results")
    print("="*50)
    trainer.plot_history(save_path='outputs/training_history.png')
    trainer.confusion_matrix_plot(save_path='outputs/confusion_matrix.png')
    
    # 6. Save Model
    print("\n" + "="*50)
    print("6. Saving Model")
    print("="*50)
    Utils.save_model(model)
    Utils.save_history(history)
    
    # 7. Test Predictions
    print("\n" + "="*50)
    print("7. Testing Predictions")
    print("="*50)
    predictor = Predictor(model)
    predictor.predict_batch(X_test[:10])
    
    print("\n" + "="*50)
    print(f"✅ Project Complete! Test Accuracy: {test_acc*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()