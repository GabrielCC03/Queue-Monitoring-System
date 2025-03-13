#**Real-Time Queue Detection and Waiting Time Estimation Using Object Tracking and Clustering**

## Project Overview  
Efficient queue management is a critical challenge in high-traffic environments. This project presents a real-time queue monitoring system that employs **YOLOv8** object detection to monitor queues, count individuals in line, and estimate waiting times dynamically.

The system processes a **video feed of a service window**, detects and counts individuals, and updates queue length estimations at fixed intervals. Clustering-based post-processing techniques are implemented to enhance accuracy by filtering out non-queue individuals.

![sample_detect](https://github.com/user-attachments/assets/1614f3a2-eb2e-4446-a5a6-1e763b3a8543)

Since this system is for general use, solutions like ROI are not effective since the angles of the camera and the shape of the lines may differ.

---

## Key Features  
- **Real-time detection** of individuals in queues using YOLOv8.  
- **Dynamic queue length estimation** at 5-second intervals.  
- **Waiting time prediction** based on queue size.  
- **Noise filtering** using DBSCAN clustering to eliminate non-queue detections.  
- **Performance evaluation** using precision, recall, MAE, and MSE metrics.  

---

### Data Collection and Preparation  
The dataset used for this project consists of a combination of benchmark datasets for people detection and real-world service queue videos.  
![DBSCAN reference](https://github.com/user-attachments/assets/1c28f252-cedd-406b-89a4-691726392208)

**Dataset Composition:**  
| Dataset | Type | Samples | Use Case |
|---------|------|---------|----------|
| People Detection | Training | 2,799 | Object detection fine-tuning |
| People Detection | Validation | 394 | Model validation |
| Service Queue Videos | Testing | 258 frames | Real-world performance evaluation |

Images and videos were manually labeled, with bounding boxes assigned to individuals in queues to train the detection model effectively.

---

### Object Detection Model: YOLOv8  
YOLOv8 was selected due to its computational efficiency and high accuracy in real-time object detection tasks. The model was fine-tuned on a **people detection dataset** to improve its performance in queue environments.  
- **Input:** Video frames extracted at fixed intervals.  
- **Output:** Bounding boxes representing detected individuals.  

---

### Preprocessing Techniques  
Preprocessing techniques were applied to optimize the input frames for detection:  

![noblur](https://github.com/user-attachments/assets/31e23002-c148-4f8f-aa15-b48b2a4bfba5)

- **Normalization:** Standardized pixel values to improve generalization.  
- **Gaussian Blur:** Reduced noise to enhance feature extraction.
![gblur](https://github.com/user-attachments/assets/ddd295f5-7983-4840-8a09-b133722400df)
- **Resizing:** Adjusted frame dimensions for optimal object detection accuracy.  

---

### Post-Processing: Queue Detection via Clustering  
Since YOLO primarily detects people without differentiating queue members from bystanders, a **clustering-based approach** was implemented to isolate queue members and ensure accurate waiting time estimations.  

**Techniques Considered:**  
- **DBSCAN Clustering:** Filtered out non-queue individuals based on spatial proximity.This clustering algorithm is helpful for finding a cluster containing the majority of the people(the queue) and just counting the people that are within that cluster. In this way, the time estimation and the count of people should not be affected by people in the frame that is not in the queue.
![plot of DBSCAN](https://github.com/user-attachments/assets/213f8df4-da1a-494e-b60b-0d2bcb085520)

- **Distance-Based Counting:** Applied bounding box proximity to determine queue membership.  

DBSCAN was ultimately selected due to its dynamic clustering capabilities without requiring predefined regions of interest (ROI).

---

## Performance Evaluation  

### Model Training and Validation  
The model was trained for **10 epochs** to fine-tune YOLOv8 for queue detection while preventing overfitting.  
**Evaluation Metrics:**  
- **Precision, Recall, and F1-Score** – Assessed detection performance.  
- **Accuracy of queue count per frame** – Measured correctness of detection.  
- **Mean Absolute Error (MAE) and Mean Squared Error (MSE)** – Evaluated waiting time estimation error.
- 
![train_results](https://github.com/user-attachments/assets/5e439684-4e19-4188-a9fc-094caa7b042b)
**Training Results:**  
| Model | MAE (± sec) | MSE | Accuracy (%) |
|--------|---------|------|----------|
| Pre-trained YOLOv8 | 1.61 | 4.27 | 15.5% |
| Fine-Tuned YOLOv8 | 1.61 | 4.27 | 15.5% |
| Fine-Tuned + Preprocessing | **1.58** | **4.21** | **16.28%** |
| Fine-Tuned + Queue Clustering | 2.84 | 15.11 | 15.12% |


