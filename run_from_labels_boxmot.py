import argparse
import numpy as np
import os
import cv2
from pathlib import Path
 
from boxmot import create_tracker, get_tracker_config
 
 
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
 
 
def main(video_path: str, dets_path: str, out_path: Path, predictions_path: Path, tracker_type: str) -> None:
    config = get_tracker_config(tracker_type)
    tracker = create_tracker(tracker_type, config, reid_weights=Path('clip_vehicleid.pt'), device=0, half=False, per_class=False)
 
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    results = []
    predictions = []
    while True:
        print(f"Processing frame {frame_idx}...", end='\r')
        ret, frame = cap.read()
        if not ret:
            break
        
        dets_array = load_detections(dets_path,frame_idx)


        # tracks = tracker.update(dets_array, frame)
        tracks, preds = tracker.update(dets_array, frame)
        
        for t in tracks:
            x1, y1, x2, y2, tid, conf, cls, _ = t
            results.append([frame_idx, int(tid), x1, y1, x2, y2, int(cls)])
        
        for pred in preds:
            x, y, w, h, assigned = pred
            predictions.append([frame_idx, x, y, w, h, assigned])
 
        frame_idx += 1
        debug = False  # Set to True to enable debug mode
        if debug:
            user_input = input("Press [n] for next frame, [q] to quit: ").strip().lower()
            if user_input == 'q':
                return  # Exit the function
        
    cap.release()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, np.array(results), fmt="%d %d %.2f %.2f %.2f %.2f %d")
    np.savetxt(predictions_path, np.array(predictions), fmt="%d %.2f %.2f %.2f %.2f %s")
    

 
 
if __name__ == "__main__":
    cam_name = '04'
    video_path = f'data/cam{cam_name}.mp4'
    dets_path = f'data/labels'
    output_path = f'output/extended_preds_main.txt'
    pred_out = f'output/extended_preds_predictions.txt'
 
    main(video_path=video_path, dets_path=dets_path, out_path=Path(output_path), predictions_path=Path(pred_out), tracker_type="botsort")
 