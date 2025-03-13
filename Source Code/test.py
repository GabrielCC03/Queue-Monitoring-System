import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
from collections import defaultdict
import xml.etree.ElementTree as ET
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def count_people_in_xml(root):

    # Find all 'object' elements
    objects = root.findall('.//object')

    # Count 'person' occurrences within 'object/name' elements
    person_count = sum(1 for obj in objects if obj.find('name').text == 'person')

    return person_count

def preprocess(test_data):

    preprocessed_images = [] 

    for image_path, image, label_data in test_data:
        # Normalize the image
        normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        # Apply Gaussian filter
        blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)

        preprocessed_images.append((image_path, blurred_image, label_data))
    
    return preprocessed_images

def read_folder(folder_path):
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Initialize an empty list to store the image and label pairs
    image_label_pairs = []

    # Iterate over the files
    for file in files:
        # Check if the file is an image
        if file.endswith('.jpg') or file.endswith('.png'):
            # Get the image path
            image_path = os.path.join(folder_path, file)

            # Get the corresponding XML file path
            xml_file = file[:-4] + '.xml'
            xml_path = os.path.join(folder_path, xml_file)

            # Read the image
            image = cv2.imread(image_path)

            # Parse the XML file
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract the label data from the XML file
            label_data = count_people_in_xml(tree)
            

            # Append the image and label pair to the list
            image_label_pairs.append((image_path, image, label_data))

    return image_label_pairs

def count_people_in_queue_dbscan(boxes):
    # Extract center points of each bounding box
    centers = np.array([(box[0] + box[2] / 2, box[1] + box[3] / 2) for box in boxes])

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=400, min_samples=2).fit(centers)  # eps and min_samples should be adjusted based on the scenario
    labels = clustering.labels_

    # Count the number of people in the queue(s)
    people = 0
    for label in labels:
        if label != -1: # If person is in a group (queue)
            people += 1

    return people

def predict_frame(frame):
    '''
    For testing
    '''

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(frame, device=device, classes=0, conf=0.3)
    result = results[0].cpu()

    boxes = result.boxes
    person_count = len(boxes)
    # Use DBSCAN to count the number of people in the queue
    # person_count = count_people_in_queue_dbscan(boxes.xywh)

    return person_count, person_count*5


def predict(test_data):

    preds = {}
    for image_name, image, label_data in test_data:

        people, ewt = predict_frame(image)
        
        # Store results
        preds[image_name] = people
    
    return preds

def evaluate(preds, test):

    '''
    Evaluate the predictions
    '''

    mae = 0
    accuracy = 0
    mse = 0

    for image_name, _, label_data in test:

        pred = preds[image_name]

        mae += abs(pred - label_data)
        accuracy += 1 if pred == label_data else 0
        mse += (pred - label_data)**2

    mae /= len(test)
    accuracy /= len(test)
    mse /= len(test)

    return mae, mse, accuracy


if __name__ == "__main__":
    # Set the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Define the Model
    model = YOLO("Source Code\\fine-tuned_yolov8n.pt")  # load a pretrained model 
    model.classes = [0] # Just detect people

    test_path = 'Dataset\\test'
    test_data = read_folder(test_path)

    """
    ptmodel = model = YOLO("yolov8n.pt")  # load a pretrained model 
    model.classes = [0] # Just detect people
    preds = predict(test_data)
    mae, mse, accuracy = evaluate(preds, test_data)
    print("Pretrained Model Evaluation:")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Accuracy: {accuracy}")
    """

    # preprocess images
    test_data = preprocess(test_data)

    preds = predict(test_data)
    mae, mse, accuracy = evaluate(preds, test_data)
    print("Fine Tuned Model Evaluation:")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Accuracy: {accuracy}")