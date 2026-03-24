"""Read the numpy and transform it to an image"""
import numpy as np
import cv2

def read_and_save(image, path_to_save):
    """Take a numpy of size lengthxwidex3, converts it to an image and saves it"""
    img_to_save = image.astype(np.uint8) if image.max() > 1.0 else (image * 255.0).astype(np.uint8)
    if len(img_to_save.shape) >= 3 and img_to_save.shape[2] >= 3:
        img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path_to_save, img_to_save)