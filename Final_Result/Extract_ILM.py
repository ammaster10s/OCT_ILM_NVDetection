import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn.metrics import f1_score

def preprocess_image(image_path, img_size=(640, 640)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")

    # Replace all white pixels (value > 250) with black
    image[image > 250] = 0

    # Detect and remove white borders
    _, thresh = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find all non-white pixels and get their coordinates
    non_white_pixels = np.where(thresh < 240)

    if non_white_pixels[0].size == 0 or non_white_pixels[1].size == 0:
        raise ValueError("No relevant pixels found in the image.")

    # Get the smallest and largest x and y coordinates and use them to create the bounding box
    y_min, y_max = np.min(non_white_pixels[0]), np.max(non_white_pixels[0])
    x_min, x_max = np.min(non_white_pixels[1]), np.max(non_white_pixels[1])

    # Crop the image to the bounding box of all non-white pixels
    image = image[y_min:y_max+1, x_min:x_max+1]

    # Noise reduction using median blur, bilateral filter, and non-local means denoising
    image_median = cv2.medianBlur(image, 5)
    image_bilateral = cv2.bilateralFilter(image_median, 9, 75, 75)
    image_denoised = cv2.fastNlMeansDenoising(image_bilateral, h=30)

    # Wavelet denoising
    coeffs = pywt.wavedec2(image_denoised, 'db1', level=2)
    coeffs[1:] = [tuple(pywt.threshold(i, value=10, mode='soft') for i in level) for level in coeffs[1:]]
    image_wavelet_denoised = pywt.waverec2(coeffs, 'db1')

    # Resize the denoised image
    image_resized = cv2.resize(image_wavelet_denoised, img_size)
    image_resized = np.expand_dims(image_resized, axis=-1)
    image_resized = np.expand_dims(image_resized, axis=0)
    image_resized = image_resized.astype('float32') / 255.0

    return image, image_resized

def binary_mask(image):
    # Convert the processed image to uint8
    image_uint8 = (image * 255).astype(np.uint8).squeeze()

    blurred_image = cv2.GaussianBlur(image_uint8, (5, 5), 0)
    
    # Apply binary thresholding
    # _, binary_image = cv2.threshold(image_uint8, 62.5, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    _, binary_image = cv2.threshold(blurred_image, 62.5, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_image

def canny_edge_detection(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

def extract_highest_y(edges):
    height, width = edges.shape
    highest_y_values = np.full(width, height)  # Initialize with maximum Y values (bottom of image)

    for x in range(width):
        column = edges[:, x]
        y_indices = np.where(column > 0)[0]
        if y_indices.size > 0:
            highest_y = y_indices.min()  # Get the highest (minimum) Y value
            highest_y_values[x] = highest_y  # Store the highest Y value

    return highest_y_values

def plot_highest_y_on_edges(edges, highest_y_values):
    height, width = edges.shape
    output_image = np.zeros_like(edges)

    for x, y in enumerate(highest_y_values):
        if y < height:  # Ensure we don't plot outside the image
            output_image[y, x] = 255  # Plot highest Y values in white

    return output_image

def extract_top_line(highest_y_values):
    # Create an image with the top line plotted
    height = np.max(highest_y_values) + 1  # Ensure the image height is sufficient
    width = len(highest_y_values)
    top_line_image = np.zeros((height, width), dtype=np.uint8)

    # Draw the top line, making it bold
    for x, y in enumerate(highest_y_values):
        if y < height:  # Ensure we don't plot outside the image
            cv2.line(top_line_image, (x, y), (x, y + 2), 255, 2)  # Plot bold line in white

    return top_line_image

def concatenate_paths(base_path, filenames):
    return [os.path.join(base_path, filename) for filename in filenames]

def process_and_visualize_images(image_paths):
    for image_path in image_paths:
        # Preprocess the image
        original_image, preprocessed_image = preprocess_image(image_path)

        # Apply binary masking
        binary_image = binary_mask(preprocessed_image)

        # Apply Canny edge detection
        edges = canny_edge_detection(binary_image)

        # Extract highest Y value edges
        highest_y_values = extract_highest_y(edges)

        # Plot highest Y values on the edges image
        highest_y_image = plot_highest_y_on_edges(edges, highest_y_values)

        # Extract and plot only the top line
        top_line_image = extract_top_line(highest_y_values)

        # Visualize the results
        plt.figure(figsize=(25, 5))
        plt.subplot(1, 5, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original Image')

        plt.subplot(1, 5, 2)
        plt.imshow(binary_image, cmap='gray')
        plt.title('Binary Mask')

        plt.subplot(1, 5, 3)
        plt.imshow(edges, cmap='gray')
        plt.title('Canny Edges')

        plt.subplot(1, 5, 4)
        plt.imshow(highest_y_image, cmap='gray')
        plt.title('Highest Y Values')

        plt.subplot(1, 5, 5)
        plt.imshow(top_line_image, cmap='gray')
        plt.title('Top Line')

        plt.show()

# Example usage
test_base_path = 'DATA_OCT'
eval_base_path = 'BOOM_ILM'

# Lists of filenames for test and evaluation datasets
image_paths = ['dr_test_1190_NV.jpg', 'img_02.jpeg', 'img_04.jpeg',
    'img_05.jpeg', 'img_06.jpeg', 'img_07.jpeg',
    'img_08.jpeg', 'img_09 (1).jpeg', 'img_10.jpeg',
    'img_11.jpeg', 'img_15.jpeg', 'img_16.jpeg', 'img_17.jpeg',
    'img_18.jpeg', 'img_19.jpeg', 'img_21.jpeg',
    'img_23.jpeg', 'img_24.jpeg', 'img_25.jpeg']

eval_filenames = ['dr_test_1190_NV.jpg', 'img_02.jpeg', 'img_04.jpeg',
    'img_05.jpeg', 'img_06.jpeg', 'img_07.jpeg',
    'img_08.jpeg', 'img_09 (1).jpeg', 'img_10.jpeg',
    'img_11.jpeg', 'img_15.jpeg', 'img_16.jpeg', 'img_17.jpeg',
    'img_18.jpeg', 'img_19.jpeg', 'img_21.jpeg',
    'img_23.jpeg', 'img_24.jpeg', 'img_25.jpeg']

test_image_paths = concatenate_paths(test_base_path, image_paths)
eval_image_paths = concatenate_paths(eval_base_path, eval_filenames)

# Combine and process both test and evaluation image paths
process_and_visualize_images(test_image_paths)
