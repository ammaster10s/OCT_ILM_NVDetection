import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import zscore
import os


def preprocess_image(image_path, img_size=(640, 640)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")

    # Replace all white pixels (value > 250) with black
    image[image > 250] = 0

    # Detect and remove white borders
    _, thresh = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)

    # Find all non-white pixels and get their coordinates
    non_white_pixels = np.where(thresh < 240)

    if non_white_pixels[0].size == 0 or non_white_pixels[1].size == 0:
        raise ValueError("No relevant pixels found in the image.")

    # Get the smallest and largest x and y coordinates and use them to create the bounding box
    y_min, y_max = np.min(non_white_pixels[0]), np.max(non_white_pixels[0])
    x_min, x_max = np.min(non_white_pixels[1]), np.max(non_white_pixels[1])

    # Crop the image to the bounding box of all non-white pixels
    image = image[y_min:y_max+1, x_min:x_max+1]

    # Noise reduction using median blur and bilateral filter
    image_median = cv2.medianBlur(image, 5)
    image_denoised = cv2.bilateralFilter(image_median, 9, 75, 75)

    # Resize the denoised image
    image_resized = cv2.resize(image_denoised, img_size)
    image_resized = np.expand_dims(image_resized, axis=-1)
    image_resized = np.expand_dims(image_resized, axis=0)
    image_resized = image_resized.astype('float32') / 255.0
    return image, image_resized

def binary_mask(image):
    # Convert the processed image to uint8
    image_uint8 = (image * 255).astype(np.uint8).squeeze()

    # Apply binary thresholding
    _, binary_image = cv2.threshold(image_uint8, 62.5, 255, cv2.THRESH_BINARY)
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

def visualize_results(original_image, binary_image, edges, highest_y_values):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.plot(range(len(highest_y_values)), highest_y_values, 'r.', markersize=2)

    plt.subplot(1, 3, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binary Mask')

    plt.subplot(1, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edges')
    plt.plot(range(len(highest_y_values)), highest_y_values, 'r.', markersize=0.7)

    plt.show()

def plot_dy_dx(highest_y_values):
    width = len(highest_y_values)
    ignore_margin = int(0.02 * width)  # 2% margin

    # Calculate the derivative dy/dx
    dy_dx = np.diff(highest_y_values)
    dy_dx = np.append(dy_dx, 0)  # Append 0 to maintain the same length

    # Calculate the mean and standard deviation of dy/dx
    mean_dy_dx = np.mean(dy_dx)
    std_dy_dx = np.std(dy_dx)

    # Define a threshold for what we consider "unusual" (e.g., mean ± 2*std)
    lower_threshold = mean_dy_dx - 0.05 * std_dy_dx
    upper_threshold = mean_dy_dx + 0.05 * std_dy_dx

    print(f"Mean dy/dx: {mean_dy_dx}")
    print(f"Standard deviation of dy/dx: {std_dy_dx}")

    # Find the indices of the slopes that are unusual
    unusual_slopes = np.where((dy_dx < lower_threshold) | (dy_dx > upper_threshold))[0]

    # Detect peaks
    peaks, _ = find_peaks(dy_dx, height=0)

    # Find the unusual slopes that are also peaks
    unusual_peaks = np.intersect1d(unusual_slopes, peaks)

    # Plot ignoring the leftmost and rightmost 2% of values
    plt.figure(figsize=(20, 5))
    plt.plot(range(ignore_margin, width - ignore_margin), dy_dx[ignore_margin:width - ignore_margin], label='dy/dx', color='blue')
    plt.xlabel("X Coordinate")
    plt.ylabel("dy/dx")
    plt.title("Derivative of Highest Y Values Across the Image (2% Margin Ignored)")

    # Highlight unusual peaks
    for peak in unusual_peaks:
        if ignore_margin <= peak < width - ignore_margin:
            plt.scatter(peak, dy_dx[peak], color='red', s=100, edgecolor='black', label='Unusual Peak' if peak == unusual_peaks[0] else "")

    plt.legend()
    plt.show()

def plot_peaks_on_image(original_image, highest_y_values):
    width = len(highest_y_values)
    ignore_margin = int(0.02 * width)  # 2% margin

    # Calculate the derivative dy/dx
    dy_dx = np.diff(highest_y_values)
    dy_dx = np.append(dy_dx, 0)  # Append 0 to maintain the same length

    # Calculate the absolute value of dy/dx
    abs_dy_dx = np.abs(dy_dx)

    # Detect peaks and troughs
    peaks, _ = find_peaks(dy_dx, height=0)
    troughs, _ = find_peaks(-dy_dx, height=0)

    # Calculate the average value of positive and negative peaks
    positive_peaks = dy_dx[peaks]
    negative_peaks = dy_dx[troughs]

    avg_positive_peaks = np.mean(positive_peaks) if len(positive_peaks) > 0 else 0
    avg_negative_peaks = np.mean(negative_peaks) if len(negative_peaks) > 0 else 0

    # Filter significant peaks
    significant_peaks = [peak for peak in peaks if dy_dx[peak] > avg_positive_peaks]
    significant_troughs = [trough for trough in troughs if dy_dx[trough] < avg_negative_peaks]

    # Get the coordinates of significant peaks and troughs
    y_coords_peaks = highest_y_values[significant_peaks]
    x_coords_peaks = significant_peaks
    y_coords_troughs = highest_y_values[significant_troughs]
    x_coords_troughs = significant_troughs

    # Plot the original image with circles around the significant peaks and troughs
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image with Significant Peaks and Troughs Circled")

    for x, y in zip(x_coords_peaks, y_coords_peaks):
        if ignore_margin <= x < width - ignore_margin:
            circle = plt.Circle((x, y), 10, color='red', fill=False, linewidth=2)
            plt.gca().add_patch(circle)

    for x, y in zip(x_coords_troughs, y_coords_troughs):
        if ignore_margin <= x < width - ignore_margin:
            circle = plt.Circle((x, y), 10, color='green', fill=False, linewidth=2)
            plt.gca().add_patch(circle)

    plt.show()

def concatenate_paths(base_path, filenames):
    return [os.path.join(base_path, filename) for filename in filenames]


def process_multiple_images(image_paths):
    for image_path in image_paths:
        try:
            # Preprocess the image
            original_image, preprocessed_image = preprocess_image(image_path)

            # Apply binary masking
            binary_image = binary_mask(preprocessed_image)

            # Apply Canny edge detection
            edges = canny_edge_detection(binary_image)

            # Extract highest Y value edges
            highest_y_values = extract_highest_y(edges)

            # Visualize the results
            visualize_results(original_image, binary_image, edges, highest_y_values)

            # Plot dy/dx of highest Y values
            plot_dy_dx(highest_y_values)

            # Plot significant peaks and troughs on the original image
            plot_peaks_on_image(original_image, highest_y_values)
        except Exception as e:
            print(f"An error occurred with image {image_path}: {e}")



# Example usage
uploaded_image_paths = ['dr_test_1190_NV.jpg', 'img_02.jpeg', 'img_04.jpeg',
    'img_05.jpeg', 'img_06.jpeg', 'img_07.jpeg',
    'img_08.jpeg', 'img_09.jpeg', 'img_10.jpeg',
    'img_11.jpeg', 'img_15.jpeg', 'img_16.jpeg',
    'img_17.jpeg', 'img_18.jpeg', 'img_19.jpeg',
    'img_21.jpeg', 'img_23.jpeg', 'img_24.jpeg',
    'img_25.jpeg', 'img_a_NV.jpeg','img_b_NV.jpeg',
    'img_c_NV.jpeg','img_d_NV.jpeg', 'img_e_NV.jpeg'
    
    ]
test_base_path = 'DATA_OCT'

test_image_paths = concatenate_paths(test_base_path, uploaded_image_paths)

process_multiple_images(test_image_paths)
