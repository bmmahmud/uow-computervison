# Studentname: B M Ashik Mahmud Email: bmam995@uowmail.edu.au
import cv2
import numpy as np
import argparse

# ----------- TASK 1 FUNCTIONS -----------

# This function rescales the given image to the target width and height while maintaining its aspect ratio
def rescale_image(image, target_width=600, target_height=480):
    height, width = image.shape[:2]
    
    # Calculate the aspect ratio
    aspect_ratio = width / height

    # Adjust dimensions based on the target width and height
    if width > target_width or height > target_height:
        if width > height:
            width = target_width
            height = int(width / aspect_ratio)
        else:
            height = target_height
            width = int(height * aspect_ratio)
            
    return cv2.resize(image, (width, height))

# Extracts keypoints using SIFT with custom parameters
def extract_keypoints(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define SIFT detector
    sift = cv2.SIFT_create(contrastThreshold=0.08, edgeThreshold=15, sigma=1.7)
    
    # Get keypoints and descriptors
    keypoints, _ = sift.detectAndCompute(gray, None)
    
    return keypoints

# Draws the keypoints on the given image
def draw_custom_keypoints(image, keypoints):
    img_with_keypoints = image.copy()
    
    for keypoint in keypoints:
        x, y = keypoint.pt
        size = keypoint.size
        
        # Draw a cross marker for each keypoint
        cv2.drawMarker(img_with_keypoints, (int(x), int(y)), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=int(size))
        
    return img_with_keypoints

# ----------- TASK 2 FUNCTIONS -----------

# Extracts SIFT descriptors for the given image
def extract_sift_descriptors(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

# Computes the chi-squared distance between two histograms
def chi_square_distance(histA, histB, eps=1e-10):
    return 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for a, b in zip(histA, histB)])

# ----------- MAIN EXECUTION -----------

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process images using SIFT")

    # Define an argument to accept one or more image file paths
    parser.add_argument('image_files', nargs='+', help='Image names you want to process')

    # Parsing the command line arguments
    args = parser.parse_args()
    image_files = args.image_files

    # If only one image is provided, execute Task-1
    if len(image_files) == 1:
        image_path = image_files[0]
        
        # Read the image
        image = cv2.imread(image_path)

        # Rescale the image
        rescaled_image = rescale_image(image)

        # Extract keypoints
        keypoints = extract_keypoints(rescaled_image)

        # Draw keypoints
        keypoints_image = draw_custom_keypoints(rescaled_image.copy(), keypoints)

        # Concatenate the original and keypoints images
        concatenated_image = np.concatenate((rescaled_image, keypoints_image), axis=1)

        # Display the result
        cv2.imshow("Original and Keypoints", concatenated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Print number of keypoints
        print(f"# of keypoints in {image_path} is {len(keypoints)}")

    # If multiple images are provided, execute Task-2
    else:
        all_descriptors = []
        keypoints_count = []

        # Rescale each image and extract SIFT keypoints and descriptors
        for image_path in image_files:
            image = cv2.imread(image_path)
            scale = 600 / image.shape[1]
            new_dim = (600, int(image.shape[0] * scale))
            image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

            keypoints, descriptors = extract_sift_descriptors(image)
            keypoints_count.append(len(keypoints))
            all_descriptors.extend(descriptors)

        all_descriptors = np.array(all_descriptors)

        K_values = [0.05, 0.10, 0.20]

        # Displaying keypoints count for each image
        print("Key-points count per image:")
        for img, count in zip(image_files, keypoints_count):
            print(f"# of keypoints in {img} is {count}")
        print("………\n")
        

        # Compute dissimilarity matrices for different K values
        for K in K_values:
            num_clusters = int(K * len(all_descriptors))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(all_descriptors, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            histograms = []

            # Construct Histograms of Visual Words for each image
            for image_path in image_files:
                image = cv2.imread(image_path)
                _, descriptors = extract_sift_descriptors(image)
                hist = np.zeros(num_clusters)
                for descriptor in descriptors:
                    distances = np.linalg.norm(centers - descriptor, axis=1)
                    label = np.argmin(distances)
                    hist[label] += 1
                histograms.append(hist / hist.sum())

            # Display dissimilarity matrix
            print(f"K={K*100}% x total number of keypoints = {num_clusters}\n")
            dissimilarity_matrix = np.zeros((len(image_files), len(image_files)))

            for i in range(len(image_files)):
                for j in range(len(image_files)):
                    dissimilarity_matrix[i, j] = chi_square_distance(histograms[i], histograms[j])

            print("Dissimilarity Matrix")
            print("\t" + "\t".join([img.split(".")[0] for img in image_files]))  # Image names as header
            for i, img in enumerate(image_files):
                row = [img.split(".")[0]]  # Starting with image name
                for j in range(len(image_files)):
                    if j <= i:  # Only filling the upper triangle of the matrix
                        row.append("{:.2f}".format(dissimilarity_matrix[i, j]))
                    else:
                        row.append("")  # Empty string for the lower triangle
                print("\t".join(row))
            print("\n")
