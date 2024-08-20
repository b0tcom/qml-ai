# Description: This script is the main entry point for the QOBA project. It initializes the models, sets up the environment, and starts the main detection and targeting loop.
import unittest
import torch
from pathlib import Path
from ultralytics import YOLO
from aimassist.aimassist import setup_environment, create_directories, initialize_models, detect_and_target
from aimassist.quantum_model import QuantumNeuralNet

if __name__ == "__main__":
    base_dir = Path('D:/Users/dgran/source/repos/qoba')
    create_directories(base_dir) 
    sct, device = setup_environment()
    facial_recognition = None  # Define the facial_recognition variable
    initialize_models(facial_recognition, YOLO)  # Initialize the models (e.g., YOLO, Quantum Neural Network, Facial Recognition)

def main():
    class QuantumNeuralNet:
        global facial_recognition, general_detection

        def __init__(self):
            self.qnn = QuantumNeuralNet()

        def test_forward(self):
            x = torch.randn(1, 4)
            output = self.qnn.forward(x)
            self.assertEqual(output.shape, torch.Size([1, 2]))
            self.assertIsInstance(output, torch.Tensor)

        def test_train_model(self):
            train_loader = [(torch.randn(1, 4), torch.tensor([0]))]
            self.qnn.train_model(train_loader, num_epochs=1, learning_rate=0.001)
            self.assertTrue(any(p.requires_grad for p in self.qnn.parameters()))

        def test_evaluate_model(self):
            test_loader = [(torch.randn(1, 4), torch.tensor([0]))]
            self.qnn.evaluate_model(test_loader)
            self.assertFalse(self.qnn.training) # Assert that the model is in evaluation mode

        if __name__ == "__main__":
            unittest.main()

    # Define the base directory of the project
    base_dir = Path('D:/Users/dgran/source/repos/qoba')
    
    # Create necessary directories for storing models, images, datasets, etc.
    create_directories(base_dir)
    
    # Set up the screen capture environment and the computational device (GPU/CPU)
    sct, device = setup_environment()
    
    # Initialize the models (e.g., YOLO, Quantum Neural Network, Facial Recognition)
    initialize_models(facial_recognition, general_detection, YOLO,)# Initialize the models (e.g., YOLO, Quantum Neural Network, Facial Recognition)
    facial_recognition, general_detection = initialize_models(base_dir, device)
    print("Models initialized successfully.")
    
    # Start the main detection and targeting loop
    detect_and_target(general_detection, sct)

def setup_quantum_model():
    quantum_modelt = QuantumNeuralNet
    return quantum_modelt

if __name__ == "__main__":
    main()