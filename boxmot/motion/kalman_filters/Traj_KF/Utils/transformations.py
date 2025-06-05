import numpy as np

def img_to_traj_domain(position, map):
    """
    Converts a position from image domain to trajectory domain using a given mapping.

    Args:
        position (np.ndarray): The position in image domain.
        map (np.ndarray): The mapping from image domain to trajectory domain.

    Returns:
        np.ndarray: The position in trajectory domain.
    """
    # # Calculate distances from `xy` to each point in the translation map
    # distances = [np.linalg.norm(np.array(xy) - np.array(point[0])) for point in map]
    
    # # Find the two closest points in the trajectory
    # min_distances = sorted(distances)[:2]
    # min_index = [distances.index(min_distance) for min_distance in min_distances]
    
    # # Ensure the two closest points are adjacent
    # if abs(min_index[0] - min_index[1]) != 1:
    #     raise ValueError("The two closest points are not adjacent")

    # # Correctly order the two closest points
    # first_index, second_index = sorted(min_index)#
    
    map_positions = np.array([p[0] for p in map])
    distances = np.linalg.norm(map_positions - position, axis=1)
    nearest_indices = np.argsort(distances).tolist()
    best = nearest_indices[0]
    next = best + 1
    prev = best - 1

    # Check bounds for next and prev
    candidates = []
    if 0 <= next < len(map):
        candidates.append(next)
    if 0 <= prev < len(map):
        candidates.append(prev)

    # Find which of next or prev is closer to the position
    if len(candidates) == 2:
        if distances[next] < distances[prev]:
            second_index = next
        else:
            second_index = prev
    elif len(candidates) == 1:
        second_index = candidates[0]
    else:
        raise ValueError("No valid adjacent segment found.")

    first_index, second_index = sorted([best, second_index])


    
    
 
 
 # this is incorrect, need to ensure that they are sequenctial win index, nor closest, this will prevent a flipping of direction, need to find closest segment, not clsoest point.

    # Get the cartesian points
    first_point = map[first_index][0]
    second_point = map[second_index][0]
    
    # Compute the perpendicular distance and the scalar projection (t)
    lat_disp, long_disp = perpendicular_distance(position, first_point, second_point)
    
    # Compute the cumulative trajectory distance (this assumes you're working with a trajectory map)
    traj_disp = map[first_index][1] + long_disp
    
    return traj_disp, lat_disp

# def traj_to_img_domain(position, map):
#     """
#     Converts a position from trajectory domain to image domain using a given mapping.

#     Args:
#         position (np.ndarray): The position in trajectory domain.
#         map (np.ndarray): The mapping from trajectory domain to image domain.

#     Returns:
#         np.ndarray: The position in image domain.
#     """
#     lat_disp, traj_disp = position
#     # Traverse through the map to find the two closest points based on trajectory distance
#     for i in range(1, len(map)):
#         first_point = map[i-1]
#         second_point = map[i]
        
#         # Check if the trajectory distance falls between the two points
#         if first_point[1] <= traj_disp <= second_point[1]:
#             # Calculate the fraction of the distance within this segment
#             frac = (traj_disp - first_point[1]) / (second_point[1] - first_point[1])
            
#             # Interpolate between the two points to get the base Cartesian coordinates
#             x = first_point[0][0] + frac * (second_point[0][0] - first_point[0][0])
#             y = first_point[0][1] + frac * (second_point[0][1] - first_point[0][1])

#             # Now adjust for the lateral displacement (lat_disp)
#             # Find the direction of the tangent (perpendicular) at the point on the trajectory
#             line_vec = np.array(second_point[0]) - np.array(first_point[0])  # vector along the line
#             line_length = np.linalg.norm(line_vec)
#             if line_length == 0:
#                 raise ValueError("The two trajectory points are the same, cannot compute direction.")

#             # Normalize the vector to get the unit direction
#             line_vec_normalized = line_vec / line_length

#             # Find the unit perpendicular vector (rotating 90 degrees counterclockwise)
#             perp_vec = np.array([-line_vec_normalized[1], line_vec_normalized[0]])

#             # Adjust the point for lateral displacement
#             x += lat_disp * perp_vec[0]
#             y += lat_disp * perp_vec[1]
            
#             return x, y
        
#     raise ValueError("The trajectory distance is outside the bounds of the translation map.")

def traj_to_img_domain(position, map):
    """
    Converts a position from trajectory domain to image domain using a given mapping.
    Allows projection beyond the final segment of the trajectory.
    
    Args:
        position (np.ndarray): (lat_disp, traj_disp) position in trajectory domain.
        map (list): List of (xy, traj_dist) tuples.

    Returns:
        np.ndarray: (x, y) position in image domain.
    """
    lat_disp, traj_disp = position

    # Handle standard interpolation within the map
    for i in range(1, len(map)):
        first_point = map[i - 1]
        second_point = map[i]

        if first_point[1] <= traj_disp <= second_point[1]:
            # Interpolate between two points
            frac = (traj_disp - first_point[1]) / (second_point[1] - first_point[1])
            base_point = np.array(first_point[0]) + frac * (np.array(second_point[0]) - np.array(first_point[0]))

            # Compute direction vector and apply lateral displacement
            line_vec = np.array(second_point[0]) - np.array(first_point[0])
            perp_vec = np.array([-line_vec[1], line_vec[0]]) / np.linalg.norm(line_vec)
            return (base_point + lat_disp * perp_vec).tolist()

    # Handle extrapolation if traj_disp is beyond the final segment
    if traj_disp > map[-1][1]:
        first_point = map[-4]
        second_point = map[-1]

        # Direction and length of last segment
        line_vec = np.array(second_point[0]) - np.array(first_point[0])
        line_length = second_point[1] - first_point[1]
        if line_length == 0:
            raise ValueError("The final two trajectory points are the same, cannot extrapolate.")

        line_vec_normalized = line_vec / line_length

        # Project beyond the last point
        extrap_disp = traj_disp - second_point[1]
        base_point = np.array(second_point[0]) + extrap_disp * line_vec_normalized

        # Lateral offset
        perp_vec = np.array([-line_vec_normalized[1], line_vec_normalized[0]])
        return (base_point + lat_disp * perp_vec).tolist()

    # (Optional) Handle extrapolation before start — not requested, but symmetric
    if traj_disp < map[0][1]:
        second_point = map[1]
        first_point = map[0]

        line_vec = np.array(second_point[0]) - np.array(first_point[0])
        line_length = second_point[1] - first_point[1]
        if line_length == 0:
            raise ValueError("The initial two trajectory points are the same, cannot extrapolate.")

        line_vec_normalized = line_vec / line_length
        extrap_disp = traj_disp - first_point[1]
        base_point = np.array(first_point[0]) + extrap_disp * line_vec_normalized

        perp_vec = np.array([-line_vec_normalized[1], line_vec_normalized[0]])
        return (base_point + lat_disp * perp_vec).tolist()

    raise ValueError("The trajectory distance is outside the bounds of the translation map.")

def create_traj_map(trajs):
    """
    Creates a mapping from image domain to trajectory domain.

    Args:
        traj_domain (np.ndarray): The trajectory domain.
        img_domain (np.ndarray): The image domain.
        trajs (list): The list of trajectories in previous fomrat.

    Returns:
        np.ndarray: The mapping from trajectory domain to image domain.
    """
    translation_map = []
    for traj in trajs:
        current_map = []
        for i, point in enumerate(traj):
            cartesian_point = (point[0], point[1])
            if i == 0:
                centre_point = 0
            else:
                distance = np.linalg.norm(np.array(traj[i-1][0]) - np.array(traj[i][0]))
                centre_point = current_map[-1][1] + distance
            current_map.append((cartesian_point, centre_point))
        translation_map.append(current_map)
    return translation_map

def perpendicular_distance(point, line_start, line_end):
    """Calculate the perpendicular distance from a point to a line segment,
    preserving the sign to indicate left/right displacement."""
    
    # Convert inputs to NumPy arrays
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    point = np.array(point)
    
    # Vector along the trajectory segment
    line_vec = line_end - line_start
    point_vec = point - line_start  # Vector from line_start to the given point

    # Squared length of the trajectory segment
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0:
        return np.linalg.norm(point - line_start), 0  # Degenerate case: segment is a point

    # Compute the projection scalar t (fraction along the segment)
    t = np.dot(point_vec, line_vec) / line_len_sq
    t = np.clip(t, 0, 1)  # Clamp to stay within the segment

    # Compute the closest point on the trajectory segment
    closest_point = line_start + t * line_vec

    # Compute the perpendicular displacement vector (from trajectory to point)
    perp_vec = point - closest_point

    # Compute a **signed** perpendicular distance
    # Get the unit perpendicular vector (90-degree counterclockwise rotation)
    perp_unit_vec = np.array([-line_vec[1], line_vec[0]])  # Perpendicular to trajectory
    perp_unit_vec /= np.linalg.norm(perp_unit_vec)  # Normalize

    # Use dot product to get signed distance
    signed_perp_dist = np.dot(perp_vec, perp_unit_vec)

    return signed_perp_dist, t