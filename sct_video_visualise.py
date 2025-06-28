import cv2
from collections import defaultdict
import ast
import numpy as np

def draw_boxes(frame, boxes, color=(0, 255, 0), thickness=2):
    for box in boxes:
        id, x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, f'ID: {id}', (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 4, color, thickness)
    return frame

def load_all_boxes(path):
    frame_boxes = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            frame, id, x1, y1, x2, y2, cls = map(float, parts)
            frame_boxes[int(frame)].append([id, x1, y1, x2, y2])
    return frame_boxes


def read_pkl(traj_dir):
    import pickle
    with open(traj_dir, 'rb') as pkl_file:
        loaded_data = pickle.load(pkl_file) 
    polygon_set = loaded_data
    return polygon_set

def read_json(json_dir):
    import json
    from shapely.wkt import loads as wkt_loads
    import numpy as np
    """
    Load from JSON, restoring:
    - Outer and inner keys as Shapely geometries
    - Trajectory values as numpy arrays
    - 'polygons' as list of Shapely geometries
    """
    with open(json_dir, 'r') as f:
        raw_data = json.load(f)

    deserialized_data = {}

    for outer_key_str, inner in raw_data.items():
        if outer_key_str == 'polygons':
            deserialized_data['polygons'] = [wkt_loads(wkt) for wkt in inner]
        else:
            outer_key = wkt_loads(outer_key_str)
            deserialized_inner = {}
            for inner_key_str, arr in inner.items():
                inner_key = wkt_loads(inner_key_str)
                deserialized_inner[inner_key] = np.array(arr)
            deserialized_data[outer_key] = deserialized_inner

    return deserialized_data

def read_traj(traj_dir):
    """
    Read trajectory data from a file, which can be in either JSON or pickle format.
    pkl seems to show some compatibility issues with numpy and boxmot versions, use json when running old numpy. 
    Json may just be the better approach anyway.
    
    Args:
        traj_dir (str): Path to the trajectory file.
    
    Returns:
        dict: The loaded trajectory data.
    """
    if traj_dir.endswith('.json'):
        return read_json(traj_dir)
    elif traj_dir.endswith('.pkl'):
        return read_pkl(traj_dir)
    else:
        raise ValueError("Unsupported file format. Use .json or .pkl.")
    
def draw_polygons(frame, polygons):
    import numpy as np
    """
    Draws polygons on the given frame.
    
    Args:
        frame (np.ndarray): The image frame on which to draw.
        polygons (list): List of Shapely polygons to draw.
    
    Returns:
        np.ndarray: The frame with polygons drawn.
    """
    for poly in polygons:
        if poly.is_empty:
            continue
        exterior = np.array(poly.exterior.coords, dtype=np.int32)
        cv2.polylines(frame, [exterior], isClosed=True, color=(255, 0, 0), thickness=2)
    return frame

def load_all_preds(pred_dir):
    """
    Load all predictions from a file.
    
    Args:
        pred_dir (str): Path to the prediction file.
    
    Returns:
        dict: A dictionary where keys are frame numbers and values are lists of prediction boxes.
    """
    frame_preds = defaultdict(list)
    with open(pred_dir, 'r') as f:
        for line in f:
            parts = line.strip().split()
            frame, x, y, w, h, assigned = parts
            frame = int(frame)
            x, y, w, h = map(float, [x, y, w, h])
            x1,y1,x2,y2 = xywh_to_xyxy([x, y, w, h])
            frame_preds[int(frame)].append([x1, y1, x2, y2, assigned])
    return frame_preds

def draw_preds(frame, preds):
    """
    Draws prediction boxes on the given frame.
    
    Args:
        frame (np.ndarray): The image frame on which to draw.
        preds (list): List of prediction boxes, each box is a list [id, x1, y1, x2, y2].
    
    Returns:
        np.ndarray: The frame with prediction boxes drawn.
    """
    for pred in preds:
        x1, y1, x2, y2, assigned = pred
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f'name: {assigned}', (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame

def xywh_to_xyxy(box):
    """
    Convert bounding box from (x, y, w, h) format to (x1, y1, x2, y2) format.
    
    Args:
        box (list): Bounding box in (x, y, w, h) format.
    
    Returns:
        list: Bounding box in (x1, y1, x2, y2) format.
    """
    x, y, w, h = map(int, box)
    return [x-w//2, y-h//2, x+w//2, y+h//2]


codec = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
cap = cv2.VideoCapture('data/cam04.mp4')
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
outputvid = cv2.VideoWriter('botsort_test.mp4', codec, 15, (width, height))
polys = read_traj('data/trajectories/cam04_traj_redo.json').pop('polygons')
predictions = True
pred_dir = 'output/botsort_new_test_preds.txt'

frame_num = 0
all_boxes = load_all_boxes('output/botsort_new_test.txt')
preds = load_all_preds(pred_dir) 
fps = 0

debugpoints = True
debug_txt = 'debug_points.txt'

while True:
    now = cv2.getTickCount()
    print(f"\rProcessing frame {frame_num} at {int(fps)} fps", end='', flush=True)
    ret, frame = cap.read()
    if not ret:
        break

    if polys is not None:
        draw_polygons(frame, polys)

    boxes = all_boxes.get(frame_num, []) # get boxes for the current frame
    if not boxes: # skipo any empty frames
        frame_num += 1
        outputvid.write(frame)
        continue
    if predictions: # if predictions are enabled, draw them
        frame_preds = preds.get(frame_num, [])
        draw_preds(frame, frame_preds)
    draw_boxes(frame, boxes) # draw boxes onto the frame


    if debugpoints:
        with open(debug_txt, 'r') as f:
            for line in f.readlines():
                point = line.strip().split(', ')
                # Draw debug points from debug_points.txt in yellow
                cv2.circle(frame, (int(float(point[0])), int(float(point[1]))), 5, (0, 255, 255), -1)

        # Read the file
        with open('34_maps.txt', 'r') as file:
            data = file.read()
        # Safely evaluate the list structure
        parsed = ast.literal_eval(data)
        # Convert to a list of NumPy arrays with dtype=object to preserve the tuple structure
        maps = []
        for segment in parsed:
            segment_array = np.array([[x, y] for ((x, y), v) in segment])
            maps.append(segment_array)
        # Draw map points from 34_maps.txt in cyan
        for m in maps:
            for point in m:
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, (255, 255, 0), -1)

    outputvid.write(frame)

    frame_num += 1
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - now) # find fps of processing

cap.release()
outputvid.release()
