import time
import matplotlib.pyplot as plt
import ctypes
import numpy as np
import torch
import pyautogui
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from aimassist.aimbot import cam as cam, get_position, get_timestamp, get_velocity
import ctypes
import numpy as np
from data_structure import AimData
import quantum_model as quantum_neuralnet

def collect_data():
    # Collect aim assist data
    position = get_position()
    velocity = get_velocity()
    timestamp = get_timestamp()
    data = AimData(position, velocity, timestamp)
    quantum_neuralnet.process_data(data)
class aimassist:

    def __init__(self, sensitivity=1.0, smooth_factor=1.0, fov=90, activation_key='shift'):
        """
        Initialize the aimbot with sensitivity, smooth factor, field of view (FOV), and activation key.

        :param sensitivity: Sensitivity multiplier for mouse movement.
        :param smooth_factor: Factor to smooth the aim movement.
        :param fov: Field of view in degrees for targeting.
        :param activation_key: Key to activate the aimbot (default: 'shift').
        """
        self.sensitivity = sensitivity
        self.smooth_factor = smooth_factor
        self.fov = fov
        self.activation_key = activation_key

        # Initialize Quantum Neural Network
        self.qnn = QuantumNeuralNet().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

from quantum_model import QuantumNeuralNet
import pyautogui
import numpy as np
import torch
import pennylane as qml
import torch.nn as nn
import torch.optim as optim
from main import main as main
# Quantum Circuit for Quantum Neural Network
dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev, interface='torch')
def quantum_conv_circuit(inputs, weights):
    # Quantum Convolutional Layer
    qml.templates.AngleEmbedding(inputs, wires=range(4))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(4))
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

class QuantumNeuralNet(nn.Module):
    """
    A quantum neural network model for binary classification.
    Attributes:
        quantum_weights (torch.nn.Parameter): Trainable weights for the quantum layer.
        fc1 (nn.Linear): First classical fully connected layer.
        fc2 (nn.Linear): Second classical fully connected layer.
        fc3 (nn.Linear): Output layer for binary classification.
        softmax (nn.Softmax): Softmax activation function.
    Methods:
        forward(x): Performs the forward pass of the model.
        train_model(train_loader, num_epochs=10, learning_rate=0.001): Trains the model using the given data.
        evaluate_model(test_loader): Evaluates the model's performance on the test data.
    """
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

class aimbot:
    def __init__(self, sensitivity=1.0, smooth_factor=1.0, fov=90, activation_key='shift'):
        """
        Initialize the aimbot with sensitivity, smooth factor, field of view (FOV), and activation key.

        :param sensitivity: Sensitivity multiplier for mouse movement.
        :param smooth_factor: Factor to smooth the aim movement.
        :param fov: Field of view in degrees for targeting.
        :param activation_key: Key to activate the aimbot (default: 'shift').
        """
        self.sensitivity = sensitivity
        self.smooth_factor = smooth_factor
        self.fov = fov
        self.activation_key = activation_key

        # Initialize Quantum Neural Network
        self.qnn = QuantumNeuralNet().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def is_key_pressed(self):
        """
        Check if the activation key is pressed.
        
        :return: Boolean indicating whether the activation key is pressed.
        """
        return pyautogui.keyDown(self.activation_key)

    def move_cursor(self, x, y):
        """
        Move the cursor to the specified (x, y) coordinates with sensitivity applied.
        
        :param x: X-coordinate on the screen.
        :param y: Y-coordinate on the screen.
        """
        pyautogui.moveTo(x * self.sensitivity, y * self.sensitivity)

    def click_target(self, button='left'):
        """
        Click the mouse at the current cursor position.
        
        :param button: Mouse button to click ('left', 'right', 'middle').
        """
        pyautogui.click(button=button)

    def aim_at_target(self, x1, y1, x2, y2):
        """
        Aim at a target defined by the bounding box (x1, y1, x2, y2).
        Moves the cursor to the center of the bounding box and clicks.
        
        :param x1: X-coordinate of the top-left corner of the bounding box.
        :param y1: Y-coordinate of the top-left corner of the bounding box.
        :param x2: X-coordinate of the bottom-right corner of the bounding box.
        :param y2: Y-coordinate of the bottom-right corner of the bounding box.
        """
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        self.move_cursor(x_center, y_center)
        self.click_target()

    def calculate_distance(self, x1, y1, x2, y2):
        """
        Calculate the Euclidean distance between two points.
        
        :param x1: X-coordinate of the first point.
        :param y1: Y-coordinate of the first point.
        :param x2: X-coordinate of the second point.
        :param y2: Y-coordinate of the second point.
        :return: The Euclidean distance between the two points.
        """
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def smooth_aim(self, current_pos, target_pos):
        """
        Smoothly move the cursor from the current position to the target position.
        
        :param current_pos: A tuple (x, y) representing the current cursor position.
        :param target_pos: A tuple (x, y) representing the target cursor position.
        """
        x_current, y_current = current_pos
        x_target, y_target = target_pos
        steps = int(self.calculate_distance(x_current, y_current, x_target, y_target) / self.smooth_factor)

        if steps == 0:
            return

        x_step = (x_target - x_current) / steps
        y_step = (y_target - y_current) / steps

        for i in range(steps):
            x_current += x_step
            y_current += y_step
            self.move_cursor(x_current, y_current)
            time.sleep(0.001)  # Small delay to simulate smooth movement

    def target_within_fov(self, detection):
        """
        Check if a detected target is within the field of view (FOV).
        
        :param detection: A dictionary containing bounding box coordinates.
        :return: Boolean indicating whether the target is within the FOV.
        """
        screen_width, screen_height = pyautogui.size()
        x_center_screen = screen_width // 2
        y_center_screen = screen_height // 2

        x1, y1, x2, y2 = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
        x_center_target = (x1 + x2) // 2
        y_center_target = (y1 + y2) // 2

        distance = self.calculate_distance(x_center_screen, y_center_screen, x_center_target, y_center_target)
        fov_radius = (self.fov / 180.0) * screen_width

        return distance <= fov_radius

    def target_closest_in_fov(self, detections):
        """
        Find the closest object within the field of view and aim at it.

        :param detections: A list of bounding boxes, where each box is represented as a dictionary
                           with keys 'xmin', 'ymin', 'xmax', 'ymax', and 'confidence'.
        """
        if not detections or not self.is_key_pressed():
            return

        closest_distance = float('inf')
        closest_object = None

        for detection in detections:
            if self.target_within_fov(detection) and detection['confidence'] > 0.5:
                x_center_target = (detection['xmin'] + detection['xmax']) // 2
                y_center_target = (detection['ymin'] + detection['ymax']) // 2
                distance = self.calculate_distance(pyautogui.size()[0] // 2, pyautogui.size()[1] // 2, x_center_target, y_center_target)
                
                if distance < closest_distance:
                    closest_distance = distance
                    closest_object = detection

        if closest_object:
            x_center_target = (closest_object['xmin'] + closest_object['xmax']) // 2
            y_center_target = (closest_object['ymin'] + closest_object['ymax']) // 2
            current_pos = pyautogui.position()
            self.smooth_aim(current_pos, (x_center_target, y_center_target))
            self.click_target()

    def train_qnn(self, train_loader, num_epochs=10, learning_rate=0.001):
        """
        Train the Quantum Neural Network using the provided training data.

        :param train_loader: DataLoader for training data.
        :param num_epochs: Number of epochs for training.
        :param learning_rate: Learning rate for the optimizer.
        """
        self.qnn.train_model(train_loader, num_epochs, learning_rate)

    def evaluate_qnn(self, test_loader):
        """
        Evaluate the Quantum Neural Network using the provided test data.

        :param test_loader: DataLoader for test data.
        """
        self.qnn.evaluate_model(test_loader)

