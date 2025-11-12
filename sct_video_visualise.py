import cv2
from collections import defaultdict
import ast
import numpy as np
import os
import json

def is_track_occluded(track_meta: dict) -> bool:
    """
    Decide occlusion truthiness from per-track metadata.
    Consider common keys; fall back to score threshold if present.
    """
    truthy_keys = ["occluded", "partial", "occl", "occl_flag", "is_occluded", "occlusion"]
    if any(bool(track_meta.get(k, False)) for k in truthy_keys):
        return True
    # Optional: score-based fallback
    score = track_meta.get("occl_score", None)
    if isinstance(score, (int, float)) and score >= 0.5:
        return True
    return False

def load_meta_occlusions(path: str):
    """
    Load your recorder output and build:
      meta_by_frame: {frame_idx: {track_id(str)}}
    Returns empty dict if file missing or unreadable.
    """
    if not os.path.exists(path):
        print(f"[meta] file not found: {path} (proceeding without occlusion colouring)")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)  # expected: list of frames with {"iteration", "tracks": {...}}
    except Exception as e:
        print(f"[meta] failed to read {path}: {e} (proceeding without occlusion colouring)")
        return {}

    meta_by_frame = {}
    # Handle either a list of frames or a dict with "frames"
    frames_iter = data if isinstance(data, list) else data.get("frames", [])
    for fr in frames_iter:
        fid = int(fr.get("frame", fr.get("iteration", -1)))
        tracks = fr.get("tracks", {}) or {}
        occ_ids = set()
        for tid, tmeta in tracks.items():
            if isinstance(tmeta, dict) and is_track_occluded(tmeta):
                occ_ids.add(str(tid))
        if fid >= 0 and occ_ids:
            meta_by_frame[fid] = occ_ids
    return meta_by_frame

def draw_boxes(frame, boxes, color=(0, 255, 0), thickness=2, occluded_ids=None):
    """
    boxes: list of [id, x1, y1, x2, y2]
    occluded_ids: optional set of track id strings that are occluded for THIS frame
    """
    occ_col = (0, 255, 255)  # yellow
    for box in boxes:
        id, x1, y1, x2, y2 = map(int, box)
        is_occ = (occluded_ids is not None) and (str(int(id)) in occluded_ids)
        col = occ_col if is_occ else color
        thick = thickness + 2 if is_occ else thickness  # thicker for visibility

        cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)
        tag = " (OCC)" if is_occ else ""
        cv2.putText(frame, f'ID: {id}{tag}', (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_PLAIN, 4, col, thick)
        centre = ( (x1 + x2)//2, (y1 + y2)//2 )
        cv2.circle(frame, centre, 5, col, -1)
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
            print(parts)
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


def load_detections(path: str,frame_idx:int) -> np.ndarray:
    
    det_path = os.path.join(path, f"img{frame_idx:06d}.txt")
    with open(det_path, 'r') as f:
        lines = f.readlines()
    det_array = np.zeros((len(lines), 6))
    for idx,line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 6:#
            print(f"Skipping line in {det_path}: {line.strip()}")
            continue
        cls, x1, y1, x2, y2, conf = map(float, parts[0:6])
        det_array[idx, 0] = x1
        det_array[idx, 1] = y1
        det_array[idx, 2] = x2
        det_array[idx, 3] = y2
        det_array[idx, 4] = conf
        det_array[idx, 5] = cls
    
    return det_array

def draw_dets(frame, dets, color=(255, 0, 0), thickness=1):
    """
    Draws detection boxes on the given frame.
    
    Args:
        frame (np.ndarray): The image frame on which to draw.
        dets (np.ndarray): Array of detections, each row is [x1, y1, x2, y2, conf, cls].
    
    Returns:
        np.ndarray: The frame with detection boxes drawn.
    """
    for det in dets:
        x1, y1, x2, y2 = map(int, det[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        centre = ( (x1 + x2)//2, (y1 + y2)//2 )
        cv2.circle(frame, centre, 3, (255, 0, 0), -1)
    return frame
 


codec = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
cap = cv2.VideoCapture('data/cam04.mp4')
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
outputvid = cv2.VideoWriter('botsort_test_depth_all.mp4', codec, 15, (width, height))
polys = read_traj('data/trajectories/cam04_traj_redo.json').pop('polygons')
predictions = False
pred_dir = 'output/botsort_normal_preds.txt'
meta_path = 'output/meta/run.json'
meta_occ_by_frame = load_meta_occlusions(meta_path)
dets_dir = 'data/labels'   # your detection labels folder


frame_num = 0
all_boxes = load_all_boxes('output/botsort_new_depth_noise_test_boxupdate.txt')
preds = load_all_preds(pred_dir) 
fps = 0

debugpoints = False
debug_txt = 'debug_points.txt'

while True:
    now = cv2.getTickCount()
    print(f"\rProcessing frame {frame_num} at {int(fps)} fps", end='', flush=True)
    ret, frame = cap.read()
    if not ret:
        break

    if polys is not None:
        draw_polygons(frame, polys)

    boxes = all_boxes.get(frame_num, [])  # get boxes for the current frame
    if not boxes:
        frame_num += 1
        outputvid.write(frame)
        continue

    # Get occluded IDs (if any) for this frame
    occluded_ids = meta_occ_by_frame.get(frame_num, set())

    if predictions:
        frame_preds = preds.get(frame_num, [])
        draw_preds(frame, frame_preds)

    # Pass occluded_ids into draw_boxes
    draw_boxes(frame, boxes, occluded_ids=occluded_ids)
    
    dets = load_detections(dets_dir, frame_num)
    draw_dets(frame, dets, color=(0, 0, 255), thickness=1)  # draw detections in blue
    
    
    # draw current frame number in top-right corner
    text = f"Frame: {frame_num}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = width - text_w - 10
    y = 10 + text_h

    # background rectangle for readability
    pad = 6
    rect_tl = (x - pad // 2, y - text_h - pad // 2)
    rect_br = (x + text_w + pad // 2, y + baseline + pad // 2)
    cv2.rectangle(frame, rect_tl, rect_br, (0, 0, 0), -1)

    # white text with a thin black outline for extra contrast
    cv2.putText(frame, text, (x, y), font, scale, (50, 50, 50), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)



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
