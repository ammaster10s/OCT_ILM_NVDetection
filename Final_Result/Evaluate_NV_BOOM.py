import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
from scipy.signal import find_peaks

def preprocess_image(image_path, img_size=(640, 640)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")

    # Replace all white pixels (value > 250) with black
    image[image > 250] = 0

    original_shape = image.shape

    # Detect and remove white borders using Otsu's thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find all non-white pixels and get their coordinates
    non_white_pixels = np.where(thresh < 255)

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

    return image, image_resized, original_shape, (y_min, y_max, x_min, x_max)

def binary_mask(image):
    # Convert the processed image to uint8
    image_uint8 = (image * 255).astype(np.uint8).squeeze()

    # Apply Otsu's binary thresholding
    _, binary_image = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
            # Sort y_indices to find the highest (smallest) y value that's not 0
            y_indices = np.sort(y_indices)
            for y in y_indices:
                if y != 0:
                    highest_y_values[x] = y
                    break
            else:
                # If all y values are 0, set to the smallest y value (0)
                highest_y_values[x] = y_indices[0]
        else:
            # Handle the case when no edges are found in the column
            if x > 0:
                highest_y_values[x] = highest_y_values[x - 1]
            else:
                highest_y_values[x] = height

    return highest_y_values

def plot_highest_y_on_edges(edges, highest_y_values):
    height, width = edges.shape
    output_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    for x, y in enumerate(highest_y_values):
        if y < height:  # Ensure we don't plot outside the image
            cv2.line(output_image, (x, height), (x, y), (0, 255, 0), 1)  # Plot lines in green

    return output_image

def plot_only_highest_y_values(edges, highest_y_values):
    height, width = edges.shape
    output_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert edges to a color image

    for x, y in enumerate(highest_y_values):
        if y < height:  # Ensure the Y value is within image bounds
            cv2.circle(output_image, (x, y), 1, (0, 255, 0), -1)  # Draw a small green dot at the highest Y value

    return output_image

def plot_highest_y_on_original(original_image, highest_y_values, scale):
    height, width = original_image.shape
    output_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)  # Convert to color image for drawing

    for x, y in enumerate(highest_y_values):
        scaled_x = int(x * scale[1])
        scaled_y = int(y * scale[0])
        if scaled_y < height and scaled_y >= 0 and scaled_x < width and scaled_x >= 0:
            cv2.circle(output_image, (scaled_x, scaled_y), 2, (0, 255, 0), -1)  # Draw a small green dot at the highest Y value

    return output_image

def detect_vertical_gaps(highest_y_values, gap_threshold=5):
    gaps = []
    start_end_points = []
    for x in range(1, len(highest_y_values)):
        if abs(highest_y_values[x] - highest_y_values[x - 1]) > gap_threshold:
            start_end_points.append((x - 1, x))
            gaps.append((x, (highest_y_values[x] + highest_y_values[x - 1]) // 2))
    return gaps, start_end_points

def plot_gaps(output_image, gaps):
    for x, y in gaps:
        cv2.circle(output_image, (x, y), 5, (255, 0, 0), 2)  # Draw a red circle around each gap

    return output_image

def calculate_tangent_vectors(highest_y_values, start_end_points):
    tangent_vectors = []
    for start, end in start_end_points:
        start_point = (start, highest_y_values[start])
        end_point = (end, highest_y_values[end])
        vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])
        tangent_vectors.append((start_point, vector))
    return tangent_vectors

def remove_border_vectors(tangent_vectors, image_width, border_margin=30):
    # Remove rising vectors close to the left border
    tangent_vectors = [vec for vec in tangent_vectors if not (vec[0][0] < border_margin and vec[1][1] < 0)]

    # Remove falling vectors close to the right border
    tangent_vectors = [vec for vec in tangent_vectors if not (vec[0][0] > image_width - border_margin and vec[1][1] > 0)]

    return tangent_vectors

def plot_tangent_vectors(image, tangent_vectors):
    for (start_x, start_y), (vec_x, vec_y) in tangent_vectors:
        end_x = start_x + vec_x
        end_y = start_y + vec_y
        cv2.arrowedLine(image, (start_x, start_y), (end_x, end_y), (255, 0, 255), 2, tipLength=0.2)
    return image

def detect_single_rising_falling_patterns(tangent_vectors):
    patterns = []
    i = 0
    while i < len(tangent_vectors) - 1:
        if tangent_vectors[i][1][1] < 0 and tangent_vectors[i + 1][1][1] > 0:
            patterns.append((tangent_vectors[i], tangent_vectors[i + 1]))
            i += 2  # Move past this pattern
        else:
            i += 1
    return patterns

def draw_pattern_circles(image, patterns, unusual_spots, scale=(1.0, 1.0)):
    detected_points = []
    detected_radii = []
    
    for rising_vector, falling_vector in patterns:
        x_coords = [rising_vector[0][0], falling_vector[0][0]]
        y_coords = [rising_vector[0][1], falling_vector[0][1]]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        center_x = int(((min_x + max_x) // 2) * scale[1])
        center_y = int(((min_y + max_y) // 2) * scale[0])
        radius = int(np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2) / 2 * max(scale))

        cv2.circle(image, (center_x, center_y), radius, (255, 0, 0), 2)
        detected_points.append((center_x, center_y))
        detected_radii.append(radius)

    for x1, y1 in unusual_spots:
        scaled_x = int(x1 * scale[1])
        scaled_y = int(y1 * scale[0])
        cv2.circle(image, (scaled_x, scaled_y), 5, (255, 255, 255), 2)
        detected_points.append((scaled_x, scaled_y))
        detected_radii.append(5)

    return image, detected_points, detected_radii



def create_binary_detected_image(detected_points, detected_radii, original_shape):
    binary_image = np.zeros(original_shape, dtype=np.uint8)

    if len(detected_points) > 0:
        for center, radius in zip(detected_points, detected_radii):
            cv2.circle(binary_image, center, radius, 255, thickness=-1)
    return binary_image

def label_connected_components(binary_image):
    # Label connected components
    num_labels, labels_im = cv2.connectedComponents(binary_image)
    return num_labels, labels_im

def check_intersection(ground_truth_labels, detected_labels_count):
    ground_truth_regions = np.unique(ground_truth_labels)[1:]  # Skip background label 0
    detected_regions_count = detected_labels_count

    true_positive = 0
    false_negative = 0
    false_positive = 0

    for gt_label in ground_truth_regions:
        gt_region = (ground_truth_labels == gt_label)
        intersection_found = False

        if detected_regions_count > 0:
            true_positive += 1
            intersection_found = True
            detected_regions_count -= 1

        if not intersection_found:
            false_negative += 1

    false_positive = detected_regions_count

    return true_positive, false_negative, false_positive

def evaluate_detection(ground_truth_image_path, detected_labels_count):
    # Load the ground truth binary image
    ground_truth_image = cv2.imread(ground_truth_image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the ground truth image is binary
    _, ground_truth_binary = cv2.threshold(ground_truth_image, 127, 255, cv2.THRESH_BINARY)

    # Label connected components
    num_labels_gt, labels_gt = label_connected_components(ground_truth_binary)

    # Check for intersections and calculate metrics
    TP, FN, FP = check_intersection(labels_gt, detected_labels_count)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score, TP, FP, FN


def detect_unusual_spots(highest_y_values, window_size=10, prominence_threshold=5, deviation_multiplier=2, slope_multiplier=2):
   
    
    unusual_spots = []
    slopes = np.diff(highest_y_values)  # Calculate the slopes

    # Calculate dynamic deviation threshold
    deviations = np.abs(np.diff(highest_y_values))
    mean_deviation = np.mean(deviations)
    std_deviation = np.std(deviations)
    deviation_threshold = mean_deviation + deviation_multiplier * std_deviation

    # Calculate dynamic slope threshold
    mean_slope = np.mean(slopes)
    std_slope = np.std(slopes)
    slope_threshold = mean_slope + slope_multiplier * std_slope

    # print(f"Deviation threshold: {deviation_threshold}")
    # print(f"Slope threshold: {slope_threshold}")

    # Calculate deviations and slopes
    for x in range(1, len(highest_y_values)):
        deviation = abs(highest_y_values[x] - highest_y_values[x - 1])
        if deviation > deviation_threshold:
            unusual_spots.append((x, highest_y_values[x]))
        
        if x >= window_size and x < len(slopes) - window_size:
            avg_slope_before = np.mean(slopes[x-window_size:x])
            avg_slope_after = np.mean(slopes[x:x+window_size])
            if abs(avg_slope_before - avg_slope_after) > slope_threshold:
                unusual_spots.append((x, highest_y_values[x]))

    # Use find_peaks to detect significant peaks and valleys
    # peaks, _ = find_peaks(highest_y_values, prominence=prominence_threshold)
    # valleys, _ = find_peaks(-highest_y_values, prominence=prominence_threshold)
    
    # for peak in peaks:
    #     unusual_spots.append((peak, highest_y_values[peak]))
    # for valley in valleys:
    #     unusual_spots.append((valley, highest_y_values[valley]))

    # Filter to keep only the most significant spots
    filtered_spots = []
    last_added_spot = None
    for spot in sorted(unusual_spots, key=lambda x: x[0]):
        if last_added_spot is None or abs(spot[0] - last_added_spot[0]) > window_size:
            filtered_spots.append(spot)
            last_added_spot = spot

    return filtered_spots




def concatenate_paths(base_path, filenames):
    return [os.path.join(base_path, filename) for filename in filenames]

def read_ground_truth_mask(filepath):
    """
    Reads the ground truth mask image from the given filepath, converts it to a binary mask,
    and draws the mask using matplotlib.
    """
    mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read the image file at {filepath}")
    
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Plotting the mask
    # plt.figure(figsize=(4, 4))
    # plt.imshow(binary_mask, cmap='gray')
    # plt.title('Ground Truth Mask')
    # plt.axis('off')
    # plt.show()
    
    return binary_mask

def process_and_visualize_images(image_paths, gt_image_paths):
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    for image_path, gt_image_path in zip(image_paths, gt_image_paths):
        # Preprocess the image
        original_image, preprocessed_image, original_shape, crop_coords = preprocess_image(image_path)

        # Apply binary masking
        binary_image = binary_mask(preprocessed_image)

        # Apply Canny edge detection
        edges = canny_edge_detection(binary_image)

        # Extract highest Y value edges
        highest_y_values = extract_highest_y(edges)

        # Plot highest Y values on the edges image
        edges_with_highest_y = plot_highest_y_on_edges(edges, highest_y_values)

        # Plot only highest Y values
        highest_y_only = plot_only_highest_y_values(edges, highest_y_values)

        # Plot only highest Y values with dots
        highest_y_with_dots = plot_only_highest_y_values(edges, highest_y_values)

        # Detect vertical gaps and get start-end points
        gaps, start_end_points = detect_vertical_gaps(highest_y_values)

        # Plot gaps on the highest Y values image
        highest_y_with_gaps = plot_gaps(highest_y_only.copy(), gaps)

        # Calculate tangent vectors
        tangent_vectors = calculate_tangent_vectors(highest_y_values, start_end_points)
        # print(f"Tangent vectors for {image_path}: {tangent_vectors}")

        # Remove border vectors
        image_width = edges.shape[1]
        tangent_vectors = remove_border_vectors(tangent_vectors, image_width)
        # print(f"Tangent vectors after border removal for {image_path}: {tangent_vectors}")

        # Detect single rising and falling patterns
        single_patterns = detect_single_rising_falling_patterns(tangent_vectors)
        # print(f"Single rising and falling patterns for {image_path}: {single_patterns}")

        # Detect unusual spots
        unusual_spots = detect_unusual_spots(highest_y_values)

        # Draw circles around the detected patterns on the original image
        scale = (original_shape[0] / 640, original_shape[1] / 640)
        original_with_pattern_circles, detected_points_rescaled, detected_radii_rescaled = draw_pattern_circles(original_image.copy(), single_patterns, unusual_spots, scale)

        # Plot highest Y values with dots on the original image
        original_with_dots = plot_highest_y_on_original(original_image.copy(), highest_y_values, scale)

        # Plot tangent vectors on a separate image without circles
        highest_y_with_vectors = plot_tangent_vectors(highest_y_only.copy(), tangent_vectors)

        # Final image with patterns and unusual spots
        final_image, detected_points, detected_radii = draw_pattern_circles(highest_y_with_vectors, single_patterns, unusual_spots, scale)

        # GroundTruth extract
        gt_mask = read_ground_truth_mask(gt_image_path)

        # Convert the "Original with Patterns" image to binary
        binary_detected_image = create_binary_detected_image(detected_points_rescaled, detected_radii_rescaled, original_shape)

        count = len(single_patterns) + len(unusual_spots)
        # Evaluate detection by comparing binary detected image with ground truth
        precision, recall, f1_score, TP, FP, FN = evaluate_detection(gt_image_path, count)

        # print(f"True Positives (TP): {TP}")
        # print(f"False Positives (FP): {FP}")
        # print(f"False Negatives (FN): {FN}")
        # print(f"Precision: {precision}")
        # print(f"Recall: {recall}")
        # print(f"F1 Score: {f1_score}")

        total_precision += precision
        total_recall += recall
        total_f1_score += f1_score

        # Visualize the results
        plt.figure(figsize=(40, 10))
        plt.subplot(1, 10, 1)
        plt.imshow(original_with_pattern_circles, cmap='gray')
        plt.title('Result using Vertical Gaps indicator \n + Outlier spot detection')

        plt.subplot(1, 10, 2)
        plt.imshow(original_with_dots, cmap='gray')
        plt.title('ILM layer detection')

        plt.subplot(1, 10, 3)
        plt.imshow(binary_detected_image, cmap='gray')
        plt.title('Detected Patterns of NVs')

        plt.subplot(1, 10, 4)
        plt.imshow(gt_mask, cmap='gray')
        plt.title('Ground Truth Mask')
        
        plt.show()
        
    num_images = len(image_paths)

    avg_precision = total_precision / num_images
    avg_recall = total_recall / num_images
    avg_f1_score = total_f1_score / num_images

    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average F1 Score: {avg_f1_score}")

# Example usage

test_filenames = [
    'dr_test_1190_NV.jpg', 'img_02.jpeg', 'img_04.jpeg',
    'img_05.jpeg', 'img_06.jpeg', 'img_07.jpeg',
    'img_08.jpeg', 'img_09.jpeg', 'img_10.jpeg',
    'img_11.jpeg', 'img_15.jpeg', 'img_16.jpeg',
    'img_17.jpeg', 'img_18.jpeg', 'img_19.jpeg',
    'img_21.jpeg', 'img_23.jpeg', 'img_24.jpeg',
    'img_25.jpeg', 'img_a_NV.jpeg','img_b_NV.jpeg',
    'img_c_NV.jpeg','img_d_NV.jpeg', 'img_e_NV.jpeg'
]
gt_filenames = [
    'dr_test_1190_NV (1)_NV.png', 'img_02 (1)_NV.png', 'img_04_NV.png',
    'img_05_NV.png', 'img_06_NV.png', 'img_07_NV.png',
    'img_08_NV.png', 'img_09 (1)_NV.png', 'img_10_NV.png',
    'img_11 (1)_NV.png', 'img_15_NV.png', 'img_16_NV.png',
    'img_17_NV.png', 'img_18_NV.png', 'img_19_NV.png',
    'img_21 (1)_NV.png', 'img_23_NV.png', 'img_24_NV.png',
    'img_25_NV.png', 'img_a_NV.png','img_b_NV.png' 
    ,'img_c_NV.png' ,'img_d_NV.png', 'img_e_NV.png'
]

test_base_path = 'DATA_OCT'

gt_base_path = 'NV_GT'

test_image_paths = concatenate_paths(test_base_path, test_filenames)
gt_image_paths = concatenate_paths(gt_base_path, gt_filenames)

process_and_visualize_images(test_image_paths, gt_image_paths)