import cv2
import numpy as np
import argparse

####### Task 1 #########
def task_one(video_file):
# Load the video
    video = cv2.VideoCapture(video_file)  

    # Initialize the background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Get the screen size
    screen_width = 1150  
    screen_height = 650  

    frame_count = 0
    while True:
        # Read a frame from the video
        ret, frame = video.read()
        frame_count += 1

        if not ret:
            break  # Exit the loop if the video is over

        # Resize the frame to VGA size
        frame = cv2.resize(frame, (640, 480))

        # Apply background subtraction to detect moving objects
        fgmask = fgbg.apply(frame)

        # Get the estimated background image
        bgimg = fgbg.getBackgroundImage()

        # Apply morphological operations to remove noise and fill the holes in the mask
        kernel = np.ones((5,5), np.uint8)
        filtered_fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        filtered_fgmask = cv2.morphologyEx(filtered_fgmask, cv2.MORPH_CLOSE, kernel)

        # Find contours of the detected objects
        contours, _ = cv2.findContours(filtered_fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize counters for different types of objects
        persons = 0
        cars = 0
        others = 0

        for contour in contours:
            # Calculate the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Classify objects based on the aspect ratio of the bounding box
            aspect_ratio = float(w)/h
            if aspect_ratio > 0.5 and aspect_ratio < 2:  # Typical aspect ratio range for persons
                persons += 1
            elif aspect_ratio > 2:  # Typical aspect ratio range for cars
                cars += 1
            else:
                others += 1

        # Create a colored version of the filtered mask
        colored_mask = cv2.bitwise_and(frame, frame, mask=filtered_fgmask)

        # Convert the binary mask to a 3 channel image
        fgmask_3channel = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)

        # Concatenate the frames
        top_row = np.concatenate((frame, bgimg), axis=1)  
        bottom_row = np.concatenate((fgmask_3channel, colored_mask), axis=1)  
        combined = np.concatenate((top_row, bottom_row), axis=0)  

        # Resize the combined frame to fit the screen
        combined = cv2.resize(combined, (screen_width, screen_height))

        # Display the combined frame
        cv2.imshow('Task-1 (bmam995)', combined)

        # Print the frame count and number of objects with their classifications
        total_objects = persons + cars + others
        print(f"Frame {frame_count:04d}: {total_objects} objects ({persons} persons, {cars} cars, {others} others)")

        # Press 'Esc' to exit the loop
        if cv2.waitKey(30) & 0xFF == 27:  # 27 is the ASCII value for 'Esc'
            break

    # Release the video capture object and close all windows
    video.release()
    cv2.destroyAllWindows()

    pass
####### Task 2 #########
def task_two(video_file):

    # Function to calculate Intersection over Union (IoU)
    def get_iou(bb1, bb2):
        # Determine the coordinates of the intersection rectangle
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])

        # If the right side of the intersection is less than the left side, or the bottom is less than the top, there is no intersection.
        # In this case, return 0.0 as the Intersection over Union (IoU).
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        return iou


    # Load the pre-trained model
    modelFile = "frozen_inference_graph.pb"
    configFile = "ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    # Open video file
    video_file = video_file
    cap = cv2.VideoCapture(video_file)

    trackers = []
    labels = {}
    frameCount = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        frame = cv2.resize(frame, (440, 280))
        
        closest_frame = frame.copy()  # Create a new frame for closest pedestrians
        closest_pedestrians = []  # Store coordinates of all detected pedestrians

        frameCount += 1
        detected_frame = frame.copy()
        labelled_frame = frame.copy()
        current_boxes = []

        for i, tracker in enumerate(trackers):
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = map(int, box)
                current_boxes.append((x, y, x+w, y+h))
                cv2.rectangle(detected_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(labelled_frame, labels[i], (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(labelled_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        
        # If the frame count is a multiple of 10, perform object detection
        if frameCount % 10 == 0:
            # Create a 4D blob from a frame
            blob = cv2.dnn.blobFromImage(
                frame, 1.0, (300, 300), (127.5, 127.5, 127.5), True, False)
            # Set the input to the pre-trained deep learning network and obtain the output predicted probabilities
            net.setInput(blob)
            # Perform a forward pass of the model to get the detections
            detections = net.forward()
            # Get the shape of the frame
            h, w = frame.shape[:2]

            # Loop through the detections
            for i in range(detections.shape[2]):
                # Get the confidence of the detection
                confidence = detections[0, 0, i, 2]
                # If the confidence is greater than 0.5
                if confidence > 0.5:
                    # Get the classId of the detection
                    classId = int(detections[0, 0, i, 1])
                    # If the classId is not 1, continue to the next detection
                    if classId != 1:
                        continue

                    # Extract bounding box coordinates from the detections
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    startX, startY, endX, endY = box.astype('int')
                    new_box = (startX, startY, endX, endY)

                    # If the new bounding box has an IoU > 0.5 with any existing box, skip it
                    if any(get_iou(new_box, existing_box) > 0.5 for existing_box in current_boxes):
                        continue

                    # Add the new bounding box to the list of closest pedestrians
                    closest_pedestrians.append(new_box) #new

                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(
                        frame, (startX, startY, endX - startX, endY - startY))
                    trackers.append(tracker)
                    labels[len(trackers)-1] = f"Person {len(trackers)}"

        # Extract top 3 closest pedestrians
        sorted_pedestrians = sorted(
            closest_pedestrians, key=lambda x: x[1], reverse=True)[:3]
        # Draw bounding boxes for the closest pedestrians
        for box in sorted_pedestrians:
            startX, startY, endX, endY = box
            cv2.rectangle(closest_frame, (startX, startY),
                        (endX, endY), (0, 0, 255), 2)

    
        # Create a blank canvas twice the size of the frame to display multiple frames together
        canvas = np.zeros((frame.shape[0]*2, frame.shape[1]*2, frame.shape[2]), dtype=frame.dtype)

        # Copy each frame to its respective position
        canvas[0:frame.shape[0], 0:frame.shape[1]] = frame
        canvas[0:frame.shape[0], frame.shape[1]:] = detected_frame
        canvas[frame.shape[0]:, 0:frame.shape[1]] = labelled_frame
        canvas[frame.shape[0]:, frame.shape[1]:] = closest_frame

        cv2.imshow('Task-2 (bmam995)', canvas)  # Display the combined frame

        # Press 'Esc' to exit the loop
        if cv2.waitKey(30) & 0xFF == 27:  # 27 is the ASCII value for 'Esc'
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()

    pass

def main():
    parser = argparse.ArgumentParser(description='Perform tasks based on given option and video file.')
    parser.add_argument('video_file', type=str, help='Path to the video file')
    parser.add_argument('-b', action='store_true', help='Perform Task One')
    parser.add_argument('-d', action='store_true', help='Perform Task Two')

    args = parser.parse_args()

    if args.b:
        task_one(args.video_file)
    elif args.d:
        task_two(args.video_file)
    else:
        print("Please provide a valid option, either '-b' or '-d'.")

if __name__ == '__main__':
    main()