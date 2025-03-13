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
    model = YOLO("yolov8n.pt")  

    # Path to your custom dataset
    data_path = "Dataset\\data.yaml"

    # Train the model
    # train_results = model.train(data=data_path, epochs=10, lr0=0.01, batch=2)
    train_results = model.train(data=data_path, epochs=10, lr0=0.01, batch=4, classes=[0])


    # Optionally, save the trained model to a file
    save_path = os.path.join('Source Code',"fine-tuned_yolov8n.pt")
    model.save(save_path)

    # Dataset https://universe.roboflow.com/yolodataset/person-dataset-mvbk4/dataset/1
