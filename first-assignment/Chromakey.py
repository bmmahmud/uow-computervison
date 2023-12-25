#bmam995 | B M Ashik Mahmud 
import cv2
import numpy as np
import argparse

# Function for the Task-1 code functionality
def first_task(args):
    # Read the image from the provided file path
    img = cv2.imread(args.imagefile)
    if img is None:
        print("Error: Could not load image.")
        return

    # Resize the image to a shape of (300, 300, 3)
    img = cv2.resize(img, (300, 300))

    # Dictionary mapping the argument values to the corresponding OpenCV color space conversion codes
    color_space_conversion = {
        "XYZ": cv2.COLOR_BGR2XYZ,
        "Lab": cv2.COLOR_BGR2Lab,
        "YCrCb": cv2.COLOR_BGR2YCrCb,
        "HSB": cv2.COLOR_BGR2HSV
    }

    # Convert the image based on chosen color space
    converted = cv2.cvtColor(img, color_space_conversion[args.color_space])

    # Split the converted image into three channels
    ch1, ch2, ch3 = cv2.split(converted)

    # Create an empty image of size 600x600 pixels with 3 channels
    frame = np.zeros((600, 600, 3), dtype=np.uint8)

    # Set sections of the new frame with the original and color converted images
    frame[0:300, 0:300] = img
    frame[0:300, 300:600] = cv2.cvtColor(ch1, cv2.COLOR_GRAY2BGR)
    frame[300:600, 0:300] = cv2.cvtColor(ch2, cv2.COLOR_GRAY2BGR)
    frame[300:600, 300:600] = cv2.cvtColor(ch3, cv2.COLOR_GRAY2BGR)

    # Display the resulting frame
    cv2.imshow("Frame: Task-1", frame)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()

# Function for the second task functionality
def second_task(args):
    # Resize helper function
    def resize_image(image, width=400, height=300):
        return cv2.resize(image, (width, height))

    # Load the green screen image and resize
    image1 = cv2.imread(args.green_screen_image_file)
    image1 = resize_image(image1)

    # Convert the image from BGR to HSV for better green color detection
    hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)

    # Define the range for detecting green color in HSV space
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Mask of detected green regions in the image
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Inverted mask to get non-green regions
    mask_object = cv2.bitwise_not(mask_green)

    # Extract the object (non-green regions) using the mask
    extracted_object = cv2.bitwise_and(image1, image1, mask=mask_object)

    # Create a white background of the same size as the input image
    white_background = np.ones_like(image1, dtype=np.uint8) * 255

    # Replace green background with white using the masks
    result_on_white = np.where(mask_object[:, :, None].astype(bool), extracted_object, white_background)

    # Load the scenic background image and resize
    scenic_image = cv2.imread(args.scenic_image_file)
    scenic_image = resize_image(scenic_image)

    # Place the extracted object onto the scenic background
    object_on_scenic = cv2.bitwise_and(scenic_image, scenic_image, mask=mask_green)
    object_on_scenic += extracted_object

    # Stack the images horizontally and vertically to create the final output
    top_row = np.hstack((image1, result_on_white))
    bottom_row = np.hstack((scenic_image, object_on_scenic))
    final_output = np.vstack((top_row, bottom_row))

    # Display the final output
    cv2.imshow('Frame: Task 2', final_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Argument parser to parse command line arguments
    parser = argparse.ArgumentParser(description="Chromakey Application")

    # Arguments for the first task functionality
    parser.add_argument("-XYZ", action="store_const", dest="color_space", const="XYZ", help="Use XYZ color space")
    parser.add_argument("-Lab", action="store_const", dest="color_space", const="Lab", help="Use Lab color space")
    parser.add_argument("-YCrCb", action="store_const", dest="color_space", const="YCrCb", help="Use YCrCb color space")
    parser.add_argument("-HSB", action="store_const", dest="color_space", const="HSB", help="Use HSB color space")

    # Positional arguments for images
    parser.add_argument("images", nargs='*', help="Path to the images. 1 image name for Task-1, 2 images names for Task-2.")

    args = parser.parse_args()

    # If no image argument is provided or more than 2 images are provided, show error and help
    if len(args.images) == 0 or len(args.images) > 2:
        print("Invalid arguments. Please provide either 1 or 2 image paths.")
        parser.print_help()
        exit()

    # If only one image is provided, and one of the color space arguments is given, use the first code
    if len(args.images) == 1 and args.color_space:
        args.imagefile = args.images[0]
        first_task(args)
    # If two images are provided, use the second code
    elif len(args.images) == 2:
        args.scenic_image_file = args.images[0]
        args.green_screen_image_file = args.images[1]
        second_task(args)
    else:
        # Handle all other scenarios and print help
        print("Invalid arguments. Please check the usage.")
        parser.print_help()