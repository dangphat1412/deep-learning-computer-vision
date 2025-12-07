"""
Image Convolution Demonstration

This script demonstrates manual convolution implementation and compares it with OpenCV's
built-in filter2D function using various kernels (blur, sharpen, edge detection, etc.).
"""

import argparse

import cv2
import numpy as np
from skimage.exposure import rescale_intensity


def convolve(image, kernel):
    """
    Apply convolution operation on an image using a given kernel.
    
    This function manually implements the convolution operation by sliding the kernel
    across the image and computing the dot product at each position.
    
    Args:
        image: Input grayscale image (2D numpy array)
        kernel: Convolution kernel/filter (2D numpy array)
        
    Returns:
        numpy.ndarray: Convolved output image
    """
    # Get the dimensions of the image and kernel
    (img_h, img_w) = image.shape[:2]
    (ker_h, ker_w) = kernel.shape[:2]

    # Calculate padding to maintain spatial dimensions
    # Padding = (kernel_size - 1) // 2 ensures output size = input size
    pad = (ker_w - 1) // 2
    
    # Add border padding to the image (replicate edge pixels)
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((img_h, img_w), dtype='float32')

    # Slide the kernel across the image from left-to-right, top-to-bottom
    for y in np.arange(pad, img_h + pad):
        for x in np.arange(pad, img_w + pad):
            # Extract the Region of Interest (ROI) centered at current (x, y)
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # Perform convolution: element-wise multiplication and sum
            k = (roi * kernel).sum()

            # Store the convolved value in the output image
            output[y - pad, x - pad] = k

    # Rescale the output to [0, 255] range and convert to uint8
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype('uint8')

    return output


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        dict: Parsed arguments
    """
    ap = argparse.ArgumentParser(description='Apply various convolution kernels to an image')
    ap.add_argument('-i', '--image', required=True,
                    help='Path to the input image')
    ap.add_argument('-s', '--save', action='store_true',
                    help='Save output images instead of displaying them')
    ap.add_argument('-o', '--output-dir', default='output',
                    help='Directory to save output images (default: output)')
    return vars(ap.parse_args())


def create_kernel_bank():
    """
    Create a collection of common convolution kernels.
    
    Returns:
        tuple: List of (kernel_name, kernel_array) tuples
    """
    # Average blurring kernels - smooth the image by averaging pixel values
    small_blur = np.ones((7, 7), dtype='float32') * (1.0 / (7 * 7))
    large_blur = np.ones((21, 21), dtype='float32') * (1.0 / (21 * 21))

    # Sharpening filter - enhances edges and details
    sharpen = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]], dtype='float32')

    # Laplacian kernel - detects edge-like regions
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype='int32')

    # Sobel x-axis kernel - detects vertical edges
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype='int32')

    # Sobel y-axis kernel - detects horizontal edges
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype='int32')

    # Emboss kernel - creates 3D embossed effect
    emboss = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]], dtype='int32')

    return (
        ('small_blur', small_blur),
        ('large_blur', large_blur),
        ('sharpen', sharpen),
        ('laplacian', laplacian),
        ('sobel_x', sobel_x),
        ('sobel_y', sobel_y),
        ('emboss', emboss)
    )


def load_and_preprocess_image(image_path):
    """
    Load an image and convert it to grayscale.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        tuple: (original_image, grayscale_image)
    """
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f'Could not load image from: {image_path}')
    
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print(f'[INFO] Loaded image: {image.shape[1]}x{image.shape[0]} pixels')
    
    return image, grey


def apply_kernels(grey_image, kernel_bank, save_mode=False, output_dir='output'):
    """
    Apply all kernels to the image and display or save results.
    
    Args:
        grey_image: Grayscale input image
        kernel_bank: Collection of (kernel_name, kernel) tuples
        save_mode: If True, save images instead of displaying
        output_dir: Directory to save output images
    """
    import os
    
    if save_mode:
        os.makedirs(output_dir, exist_ok=True)
        print(f'[INFO] Saving results to: {output_dir}/')
    
    for (kernel_name, kernel) in kernel_bank:
        print(f'[INFO] Applying {kernel_name} kernel...')
        
        # Apply kernel using custom convolution
        convolve_output = convolve(grey_image, kernel)
        
        # Apply kernel using OpenCV's built-in function
        opencv_output = cv2.filter2D(grey_image, -1, kernel)
        
        if save_mode:
            # Save output images
            cv2.imwrite(f'{output_dir}/{kernel_name}_custom.png', convolve_output)
            cv2.imwrite(f'{output_dir}/{kernel_name}_opencv.png', opencv_output)
            print(f'[INFO] Saved {kernel_name} results')
        else:
            # Display output images
            cv2.imshow('Original', grey_image)
            cv2.imshow(f'{kernel_name} - custom convolve', convolve_output)
            cv2.imshow(f'{kernel_name} - OpenCV filter2D', opencv_output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    """
    Main execution function.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Load and preprocess image
    original, grey = load_and_preprocess_image(args['image'])
    
    # Create kernel bank
    kernel_bank = create_kernel_bank()
    print(f'[INFO] Created {len(kernel_bank)} convolution kernels')
    
    # Apply all kernels
    apply_kernels(grey, kernel_bank, args['save'], args['output_dir'])
    
    if not args['save']:
        print('[INFO] Press any key in the image window to continue to the next kernel')


if __name__ == '__main__':
    main()
    