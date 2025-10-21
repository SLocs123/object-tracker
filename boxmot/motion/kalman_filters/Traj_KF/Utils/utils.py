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

def is_within(xy, polygons):
    from shapely.geometry import Point
    """
    Determine if a given point lies within any of the provided polygons.
    
    Args:
        xy (tuple): Coordinates of the point as (x, y).
        polygons (list): A list of shapely.geometry.Polygon objects.
    
    Returns:
        shapely.geometry.Polygon or None: The polygon containing the point, 
        or None if the point is not within any polygon.
    """
    point = Point(xy[0], xy[1])
    for polygon in polygons:
        if polygon.contains(point):
            return polygon
    return None

def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    m = n // 2
    if n % 2:
        return xs[m]
    return 0.5 * (xs[m-1] + xs[m])

def _mad(xs, eps=1e-9):
    # Median Absolute Deviation (robust scale). No Gaussian factor — we just need a stable scale.
    if not xs:
        return 1.0
    med = _median(xs)
    return max(_median([abs(x - med) for x in xs]), eps)


def _recent(items, k):
    return items[-k:] if len(items) > k else items