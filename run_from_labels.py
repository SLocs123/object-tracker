from objecttracker.tracker import Tracker
from objecttracker.config import ObjectTrackerConfig
from objecttracker.config import RedisConfig
from pathlib import Path
import cv2
import numpy as np

def get_directory(dir, frame):
    return f"{dir}/img{frame:06d}.txt"

def get_labels_from_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        labels = []
        for line in lines:
            label = line.strip().split(" ")
            labels.append(label)
    return labels

redis_config = RedisConfig(stream_id="default_stream")  # Replace "default_stream" with your actual stream ID if needed
config = ObjectTrackerConfig(redis=redis_config)
tracker = Tracker(config)
directory = 'data/labels'
output_directory = 'data/output'

video_path = 'data/cam04.mp4'
cap = cv2.VideoCapture(video_path)
frame_num = 0
while True:
    print(f"Processing frame {frame_num}")
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break
    label = get_labels_from_file(get_directory(directory, frame_num))
    label = np.array(label)
    
    output, time = tracker.get(label, frame)

    with open(output_directory + '/test.txt', 'a') as f:
        for i in range(len(output)):
            f.write(f"{output[i][0]} {output[i][1]} {output[i][2]} {output[i][3]} {output[i][4]} {output[i][5]} {time}\n")

    frame+=1




    
#     # Create a dummy input proto
#     input_proto = {
#         'image_path': image_path,
#         'detections': []
#     }
    
#     # Parse detections from the file
#     for line in lines[1:]:
#         parts = line.strip().split(',')
#         if len(parts) < 5:
#             continue  # Skip invalid lines
#         detection = {
#             'class_id': int(parts[0]),
#             'x1': float(parts[1]),
#             'y1': float(parts[2]),
#             'x2': float(parts[3]),
#             'y2': float(parts[4])
#         }
#         input_proto['detections'].append(detection)
    
#     # Initialize the tracker with a dummy config
#     config = Tracker.config.model_config()
#     tracker = Tracker(config)
    
#     # Call the tracker with the input proto
#     output = tracker(input_proto)
    
#     print(f'Processed {file.name}, output: {output}')

