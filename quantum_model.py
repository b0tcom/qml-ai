from aimassist import threading
import pennylane as qml
from pennylane import qml as QuantumNeuralNet
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Quantum Circuit for Quantum Neural Network
dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev, interface='torch')
def quantum_conv_circuit(inputs, weights):
    # Quantum Convolutional Layer
    qml.templates.AngleEmbedding(inputs, wires=range(4))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(4))
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# Define the hybrid quantum-classical neural network
class QuantumNeuralNet(nn.Module):
    def __init__(self):
        super(QuantumNeuralNet, self).__init__()
        # Define quantum layer with trainable weights
        self.quantum_weights = torch.nn.Parameter(
            0.01 * torch.randn(3, 4, 3))  # 3 layers, 4 wires, 3 parameters per wire
        # Define classical layers
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Output layer: 2 neurons (binary classification)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Ensure input x is on the correct device (CPU/GPU)
        x = x.to(self.quantum_weights.device)
        # Quantum convolutional layer forward pass
        q_out = quantum_conv_circuit(x, self.quantum_weights)
        # Convert quantum output to tensor
        q_out = torch.tensor(q_out, device=x.device)
        # Classical forward pass
        x = torch.relu(self.fc1(q_out))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def train_model(self, train_loader, num_epochs=10, learning_rate=0.001):
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # Get inputs and labels
                inputs, labels = data
                inputs, labels = inputs.to(self.quantum_weights.device), labels.to(self.quantum_weights.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()
                if i % 10 == 9:  # Print every 10 batches
                    print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.3f}')
                    running_loss = 0.0

        print('Finished Training')

    def evaluate_model(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(self.quantum_weights.device), labels.to(self.quantum_weights.device)
                outputs = self.forward(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy: {100 * correct / total:.2f}%')

def get_data_loaders(batch_size=32):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.FakeData(transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = datasets.FakeData(transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    # Initialize Quantum Neural Network
    qnn = QuantumNeuralNet()

    # Get data loaders
    train_loader, test_loader = get_data_loaders()

    # Train the model
    qnn.train_model(train_loader)

    # Evaluate the model
    qnn.evaluate_model(test_loader)
