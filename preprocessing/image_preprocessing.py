import numpy as np
from PIL import Image

def preprocess_image_dummy(image: Image.Image) -> np.ndarray:
    """
    Dummy preprocessing for Phase 1.
    Resizes image to 224x224 and returns a normalized numpy array.
    """
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return img_array
