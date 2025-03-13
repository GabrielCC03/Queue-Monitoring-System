'''
Task 3 Devise, implement, validate, and enhance the algorithm capable of
accurately counting the customers in the queue. (20 marks)
'''

from ultralytics import YOLO, utils
import os
import torch

if __name__ == "__main__":

    torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load a pretrained model
    model = YOLO("Source Code\\fine-tuned_yolov8n.pt")  

    # Path to your custom dataset
    data_path = "Dataset\\data.yaml"

    # Validate the model
    val_results = model.val(data='data.yaml', batch=4, classes=[0])
