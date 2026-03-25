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
    
    h, w = img.shape[:2]
    
    # Downsample si l'img est trop grande (> 512)
    if h > 512 or w > 512:
        scale = 256 / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w
        
    # Center crop au lieu d'un crop en haut à gauche pour garder le sujet centré
    y_start = max(0, (h - 256) // 2)
    x_start = max(0, (w - 256) // 2)
    img = img[y_start:y_start+256, x_start:x_start+256, :]
    return img