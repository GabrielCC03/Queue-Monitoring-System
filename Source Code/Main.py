import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN

def preprocess(img):

    # Normalize the image
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    # Apply Gaussian filter
    blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)

    return blurred_image


def open_cap(filename):

    video_path = os.path.join('source', filename)

    if not os.path.isfile(video_path):
        print("Invalid video path. Please try again.")
        return None, None

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join('out', filename[:-4] + 'DBSCAN_output.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    return cap, out

def count_people_in_queue_dbscan(boxes):
    
    # Extract center points of each bounding box
    centers = np.array([(box[0] + box[2] / 2, box[1] + box[3] / 2) for box in boxes])

    if len(centers) == 0:
        return 0

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=400, min_samples=2).fit(centers)  # eps and min_samples should be adjusted based on the scenario
    labels = clustering.labels_

    # Count the number of people in the queue(s)
    people = 0
    for label in labels:
        if label != -1: # If person is in a group (queue)
            people += 1

    return people


def calculate_wait_time(path):

    if path is None:
        print("No video provided")
        return

    # Open the video
    cap, out = open_cap(path)

    if cap is None:
        return

    print("Processing the video...")

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform inference
            results = model.track(frame, classes=0, conf=0.3, verbose=False)
            result = results[0].cpu()

            # Visualize the results on the frame
            annotated_frame = result.plot()

            # Get the bounding boxes
            boxes = result.boxes
            
            # Use DBSCAN to count the number of people in the queue
            person_count = count_people_in_queue_dbscan(boxes.xywh)

            # Update the person count based on the number of boxes
            # person_count = len(boxes)
                
            # Draw person count and Estimated time on frame
            cv2.putText(annotated_frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Estimated Waiting Time: {person_count*5} seconds", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Write the frame to the output video file
            store = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            out.write(store)

            # print(f"Person Count: {person_count}")
            # print(f"Estimated Waiting Time: {person_count*5} seconds")
            continue

        cap.release()
        out.release()
        print("Video processing complete.")

def cli():
    while True:
        video_path = input("Enter the path of a video file (or 'quit' or 'q' to exit): ")
        
        if video_path == 'quit' or video_path == 'q':
            print("Exiting the program...")
            break
        
        # Calculate the waiting time
        frames = calculate_wait_time(video_path)

if __name__ == "__main__":

    # Define the Model
    model = YOLO("Source Code\\fine-tuned_yolov8n.pt")  
    cli()