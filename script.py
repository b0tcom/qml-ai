import numpy as np
from mss import mss
from PIL import Image
from aimassist.aimbot import Aimbot
from aimassist.utils import check_cuda

def setup_environment():
    """
    Set up the environment for screen capture and device configuration.
    
    :return: sct (screen capture tool instance), device (CUDA or CPU device)
    """
    # Define the screen area to capture (modify these values as needed)
    mon = {'top': 0, 'left': 0, 'width': 500, 'height': 500}
    
    # Initialize the screen capture tool (mss)
    sct = mss()
    
    # Check if CUDA is available, otherwise use CPU
    device = check_cuda()
    
    return sct, device

def capture_screen(sct, mon):
    """
    Capture the screen area defined by 'mon' using the mss tool.
    
    :param sct: mss screen capture instance.
    :param mon: Dictionary defining the screen area to capture.
    :return: Captured frame as a NumPy array.
    """
    # Capture the screen region
    screenshot = sct.grab(mon)
    
    # Convert the screenshot to an RGB image
    img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
    
    # Convert the image to a NumPy array for processing
    return np.array(img)

def detect_and_target(model_wrapper, sct):
    """
    Continuously capture screen, detect targets, and aim at them using the provided model.
    
    :param model_wrapper: A model wrapper (e.g., YOLO) used for detecting objects.
    :param sct: mss screen capture instance.
    """
    # Define the monitor region for capturing the screen
    mon = {'top': 0, 'left': 0, 'width': 500, 'height': 500}
    
    # Initialize the aimbot instance with desired parameters
    aimbot = Aimbot(sensitivity=1.5, smooth_factor=1.0, fov=90, activation_key='shift')
    
    while True:
        # Capture the current frame
        frame = capture_screen(sct, mon)
        
        # Use the model to detect objects in the frame
        results = model_wrapper.detect(frame)
        
        # Extract detection results and target the closest object within the field of view (FOV)
        detections = results.pandas().xyxy[0].to_dict(orient="records")
        aimbot.target_closest_in_fov(detections)
