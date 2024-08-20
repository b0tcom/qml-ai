# aimassist/screen_capture.py

import numpy as np
from mss import mss
from PIL import Image

def capture_screen(mon):
    sct = mss()
    screenshot = sct.grab(mon)
    img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
    return np.array(img)
if __name__ == "__main__":
    mon = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
    img = capture_screen(mon)
    print(img.shape)
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
    plt