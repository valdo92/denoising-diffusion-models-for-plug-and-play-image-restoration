"""load the images"""
import numpy as np
import cv2


def load_image(path_to_image):
    """
    Load the image from path_to_image
    return: Numpy float32, HWC, BGR, [0,1]
    """
    img = cv2.imread(path_to_image, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_GRAYSCALE
    if img is not None and len(img.shape) >= 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img