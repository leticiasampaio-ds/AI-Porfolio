import cv2
import numpy as np
from pathlib import Path

class ImagePreprocessor:
    """
    Preprocesses and saves images with:
        1. Convert to Grayscale.
        2. Denoising Filter.
        3. Contrast Limited Adaptive Histogram Equalization (CLAHE).
        4. Normalization & Standardization.

    Args:
        image_dir_path (Path): Source directory containing subfolders of labeled images.
        save_root_path (Path): Destination folder to save processed images.
    """
    def __init__(self, image_dir_path: Path, save_root_path: Path) -> None:
        self.image_dir_path = image_dir_path
        self.save_root_path = save_root_path

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the preprocessing steps to a single image.
        
        Args:
            image (np.ndarray): Original image in BGR format.
        
        Returns:
            np.ndarray: Processed image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # The parameters values are empirical defaults
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Normalization and Standardization
        norm = enhanced.astype(np.float32) / 255.0
        mean, std = norm.mean(), norm.std()
        standardized = (norm - mean) / (std + 1e-8) # 1e-8 constant to avoid division by zero

        return standardized

    def save_image(self, image: np.ndarray, label: str, filename: str) -> None:
        """
        Saves a preprocessed image to its corresponding label directory.

        Args:
            image (np.ndarray): Preprocessed image (float32).
            label (str): Label name / class folder.
            filename (str): Name of the file to save.
        """
        save_dir = self.save_root_path / label
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename

        # Re-scaling a standardized image back into the valid pixel range [0, 255], so it can be saved as a proper image file (.jpg)
        image_to_save = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        cv2.imwrite(str(save_path), image_to_save)

    def run(self):
        """
        Executes the preprocessing pipeline on all images in the dataset.
        """
        pass