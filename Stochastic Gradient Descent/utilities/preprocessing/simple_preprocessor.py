"""Simple image preprocessor for resizing images to fixed dimensions."""

import cv2
import numpy as np


class SimplePreprocessor:
    """A simple image preprocessor that resizes images to fixed dimensions.
    
    This preprocessor resizes images to a specified width and height,
    ignoring the original aspect ratio. Useful for preparing images
    for neural networks that require fixed input dimensions.
    """

    def __init__(
        self, 
        width: int, 
        height: int, 
        interpolation: int = cv2.INTER_AREA
    ) -> None:
        """Initialize the preprocessor with target dimensions.
        
        Args:
            width: Target width for resized images.
            height: Target height for resized images.
            interpolation: OpenCV interpolation algorithm to use during resizing.
                Defaults to cv2.INTER_AREA, which is best for shrinking images.
                Common options:
                    - cv2.INTER_AREA: Best for shrinking
                    - cv2.INTER_LINEAR: Bilinear interpolation (fast)
                    - cv2.INTER_CUBIC: Bicubic interpolation (slower, higher quality)
                    - cv2.INTER_LANCZOS4: Lanczos interpolation (highest quality)
        """
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize the image to fixed dimensions, ignoring aspect ratio.
        
        Args:
            image: Input image as a NumPy array (typically from cv2.imread).
        
        Returns:
            Resized image as a NumPy array with dimensions (height, width, channels).
        
        Raises:
            ValueError: If the input image is None or empty.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is None or empty")
        
        return cv2.resize(
            image, 
            (self.width, self.height), 
            interpolation=self.interpolation
        )