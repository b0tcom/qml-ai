import importlib
import os
from json.tool import main
from pathlib import Path
from tkinter import Image
import cv2
import mss
import numpy as np
import torch
from matplotlib.font_manager import weight_dict
from pynput import keyboard
from termcolor import colored
from ultralytics import YOLO
# __init__.py for aimassist module

def is_prime(n: int) -> bool:
    """
    Return True if n is prime, False otherwise.
    """
    if n < 2:
        return False
    elif n < 4:
        return True
    elif n % 2 == 0 or n % 3 == 0:
        return False
    else:
        for i in range(5, int(n**0.5)+1, 6):
            if n % i == 0 or n % (i+2) == 0:
                return False
        return True

__all__ = [
    'FacialRecognition',
    'YOLOWrapper',
    'capture_screen',
    'target_object'
]
# Initialize the camera
cam = cv2.VideoCapture(0)

# Placeholder for facial recognition
facial_recognition = None

def Define_paths():
    dataset_dir = Path('d:\\Users\\dgran\\source\\repos\\qoba\\lib\\data')
    base_dir = Path('d:\\Users\\dgran\\source\\repos\\qoba')

    # Create the directories if they don't exist
    os.makedirs(base_dir / 'lib', exist_ok=True)
    os.makedirs(weight_dict, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    images_dir = base_dir / 'lib' / 'images'
    os.makedirs(base_dir / 'lib' / 'weights', exist_ok=True)

    # Define screen capture settings
    mon = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
    sct = mss()

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define a threshold for confidence in pseudo-labeling
    confidence_threshold = 0.5

    # Initialize the model
    class YOLOWrapper:
        def __init__(self, model_path):
            self.model = YOLO(model_path)
            self.model.to(device)
            self.model.eval()

        def detect(self, frame):
            # Run the model on the frame and return results
            results = self.model(frame)
            return results

    # Capture screen interpolation fps
    def capture_screen():
        """Capture a screenshot and return it as a NumPy array."""
        screenshot = sct.grab(mon)
        img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
        img = np.array(img)
        return img

    # Detect and target function
    def detect_and_target(model_wrapper):
        # Capture the screen
        frame = capture_screen()

        # Detect objects in the frame
        results = model_wrapper.detect(frame)

        # Process the results
        for result in results.pandas().xyxy[0].iterrows():
            _, data = result
            x1, y1, x2, y2 = data['xmin'], data['ymin'], data['xmax'], data['ymax']
            confidence = data['confidence']

            # Check if the confidence is above the threshold
            if confidence.item() > confidence_threshold:
                # Target the object
                target_object(x1, y1, x2, y2)

    # Target object function
    def target_object(x1, y1, x2, y2):
        # Calculate the center of the object
        def calculate_center(x1, y1):
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2
            # Add more logic here if needed
        calculate_center(x1, y1)

    # Initialize models
    base_dir = Path('d:\\Users\\dgran\\source\\repos\\qoba')
    weights_dir = base_dir / 'lib' / 'weights'
    images_dir = base_dir / 'lib' / 'images'
    dataset_dir = Path('d:\\user\\source\\repos\\qoba\\lib\\dataset')

    keras_model_path = weights_dir / 'facenet_keras_weights.h5'
    yolo_model_path = weights_dir / 'yolov8x-oiv7.pt'

    facial_recognition = facial_recognition(model_path=str(keras_model_path))

    if __name__ == "__main__":
        main()
        # Create the directories if they don't exist
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(base_dir / 'lib' / 'weights', exist_ok=True)

        # Define screen capture settings
        mon = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        sct = mss()

        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define a threshold for confidence in pseudo-labeling
        confidence_threshold = 0.5

        # Initialize the model
        class YOLOWrapper:
            def __init__(self, model_path):
                self.model = YOLO(model_path)
                self.model.to(device)
                self.model.eval()

            def detect(self, frame):
                # Run the model on the frame and return results
                results = self.model(frame)
                return results

        # Capture screen interpolation fps
        def capture_screen():
            """Capture a screenshot and return it as a NumPy array."""
            screenshot = sct.grab(mon)
            img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
            img = np.array(img)
            return img

        # Detect and target function
        def detect_and_target(model_wrapper):
            # Capture the screen
            frame = capture_screen()

            # Detect objects in the frame
            results = model_wrapper.detect(frame)

            # Process the results
            for result in results.pandas().xyxy[0].iterrows():
                _, data = result
                x1, y1, x2, y2 = data['xmin'], data['ymin'], data['xmax'], data['ymax']
                label, confidence = data['name'], data['confidence']

                # Check if the confidence is above the threshold
                if confidence > confidence_threshold:
                    # Target the object
                    target_object(x1, y1, x2, y2)

        def target_object(x1, y1, x2, y2):
            # Check if the coordinates are valid
            if x1 is None or y1 is None or x2 is None or y2 is None:
                return

            # Calculate the center of the object
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2
            # Add more logic here if needed

    def main():
        global facial_recognition, general_detection
        facial_recognition = facial_recognition(model_path=str(keras_model_path))
        general_detection = facial_recognition(model_path=str(yolo_model_path))

    def on_key_release(key):
        global facial_recognition, general_detection
        if key == keyboard.Key.f1:
            facial_recognition = facial_recognition(
                model_path=str('D:\\Users\\dgran\\source\\repos\\qoba\\lib\\weights\\facenet_keras_weights.h5'))

    # Example usage
    if __name__ == "__main__":
        main()
        facial_recognition = facial_recognition(
            model_path=str('D:\\Users\\dgran\\source\\repos\\qoba\\lib\\weights\\facenet_keras_weights.h5'))
        status = colored('Facial recognition enabled', 'green')
        print(status)

    def main():
        global facial_recognition, general_detection
        facial_recognition = facial_recognition(model_path=str(keras_model_path))
        general_detection = facial_recognition(model_path=str(yolo_model_path))
        return False

    if __name__ == "__main__":
        main()

    def initialize_models(__name__, keras_model_path, yolo_model_path, facial_recognition, general_detection):
        def main():
            global facial_recognition, general_detection
            facial_recognition = facial_recognition(model_path=str(keras_model_path))
            general_detection = facial_recognition(model_path=str(yolo_model_path))

        if __name__ == "__main__":
            main()

    def ey():
        general_detection = facial_recognition(model_path=str(yolo_model_path))
        status = colored('General detection enabled', 'blue')
        print(status)
        initialize_models(__name__, keras_model_path, yolo_model_path, facial_recognition, general_detection)

    if ey == keyboard.Key.f2:
        general_detection = facial_recognition(model_path=str(yolo_model_path))
        status = colored('General detection enabled', 'blue')
        print(status)
        initialize_models(__name__, keras_model_path, yolo_model_path, facial_recognition, general_detection)

    def on_key_press(key):
        os.system('cls' if os.name == 'nt' else 'clear')
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
        print(colored('''  ____                         ____                

 / ___| __ _ _ __ ___   ___   / ___|  ___ _ ____   _____ _ __ 
| |  _ / _` | '_ ` _ \ / _ \  \___ \ / _ \ '__\ \ / / _ \ '__|
| |_| | (_| | | | | | |  __/   ___) |  __/ |   \ V /  __/ |   
 \____|\__,_|_| |_| |_|\___|  |____/ \___|_|    \_/ \___|_|   
                  (Quantum AI Tool)''', 'purple'))

    # Prompt user for self-training
    user_input = input("Do you want to perform selective self-training? (yes/no): ").strip().lower()

    if user_input == 'yes':
        # Dynamically import and run the self-training script
        script_path = Path('d:\\user\\source\\repo\\qoba\\lib\\weights\\script.py')
        spec = importlib.util.spec_from_file_location("script", script_path)
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        script.train_model()  # Assuming script.py has a train_model function to start self-training

    elif user_input == 'no':
        # Initialize and run the object detection model from aimassist.py
        detector = facial_recognition(model_path=str(Path('D:\\Users\\dgran\\source\\repos\\qoba\\mobilenet_ssd_deploy.caffemodel\\mobilenet_ssd_deploy_prototxt.txt')))
        detector.run_detection()

    def pywinget_win():
        """
        This function is used to define the 'cam' variable.
        """
        pass

        cv2._InputArray_STD_VECTOR_CUDA_GPU_MAT
        pass

        cv2.WINDOW_GUI_NORMAL(0)
        from pywinget import threading  # type: ignore
        import capture_screen as main_function  # type: ignore