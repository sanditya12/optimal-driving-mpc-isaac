import numpy as np
import math
 

def get_distance(point1, point2) -> float:
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5


def create_line(point:list, rad_angle: int, length: int)-> list:
    x_new = point[0] + length * np.cos(rad_angle)
    y_new = point[1] + length * np.sin(rad_angle)
    return [point[0], point[1]],[x_new, y_new] 


 
def is_intersecting_with_line(line1: list, line2: list):
    # Unpack line segments
    p1, p2 = line1
    p3, p4 = line2
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
 
    # Compute determinants
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
 
    # If the determinant is zero, the lines are parallel (or coincident) and won't have a unique intersection point
    if det == 0:
        # print("x1= ", x1)
        # print("x2= ", x2)
        # print("y3= ", y3)
        # print("y4= ", y4)
        # print("y1= ", y1)
        # print("y2= ", y2)
        # print("x3= ", x3)
        # print("x4= ", x4)
        return None
 
    # Else, compute the intersection point using the determinants
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det


    # [19.149312749687212, 2.8826650222311896],
    # [19.35982767266464, 3.603663067032734],

    # no_intersections
    # [3.79036056] [0.79054436] [32.34874296] [9.97851445]
 
    # Check if the intersection point lies within both line segments
    tolerance = 1e-6
    if (min(x1, x2)-tolerance <= px <= max(x1, x2)+tolerance) and (min(y1, y2)-tolerance <= py <= max(y1, y2)+tolerance) \
            and (min(x3, x4)-tolerance <= px <= max(x3, x4)+tolerance) and (min(y3, y4) -tolerance<= py <= max(y3, y4)+tolerance):
        # print(line2)
        return [px, py]
 
    # print("x1= ", x1)
    # print("x2= ", x2)
    # print("y3= ", y3)
    # print("y4= ", y4)
    # print("y1= ", y1)
    # print("y2= ", y2)
    # print("x3= ", x3)
    # print("x4= ", x4)
    return (None)
 
def convert_points_to_segments(points):
    segments = []
    for i in range(len(points) - 1):  # Stop iteration one step before the last element
        segments.append((points[i], points[i + 1]))
    
    # Close the loop by connecting the last point to the first
    segments.append((points[-1], points[0]))
    return segments

def is_intersecting_with_points(line, points):
    segments = convert_points_to_segments(points)
    closest_dist = math.inf
    closest_point = None
    for segment in segments:
        point = is_intersecting_with_line(line, segment)
        if point:
            dist_to_point =get_distance(line[0], point) 
            if dist_to_point < closest_dist:
                closest_point = point
                closest_dist = dist_to_point
    if closest_point:
        return closest_point
    return None


def is_circle_intersecting_with_line(circle_center, circle_radius, point1, point2):
    """Check if a circle is intersecting a finite line segment defined by two points."""
 
    x0, y0 = circle_center
    x1, y1 = point1
    x2, y2 = point2
 
    # Calculate the perpendicular distance from the circle center to the line
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    distance = numerator / denominator
    tolerance = 1e-1
    # Check if the line segment intersects the circle when extended infinitely
    if distance <= circle_radius:
        # Check if either of the line segment endpoints is within the circle
        if (
            (x1 - x0) ** 2 + (y1 - y0) ** 2 <= circle_radius ** 2 +tolerance or
            (x2 - x0) ** 2 + (y2 - y0) ** 2 <= circle_radius ** 2 +tolerance
        ):
            return True
 
    return False

def convert_points_to_segment_noloop(points):
    segments = []
    for i in range(len(points) - 1):  # Stop iteration one step before the last element
        segments.append((points[i], points[i + 1]))
    return segments

def is_circle_intersecting_with_points(circle_center, circle_radius, points):
    segments = convert_points_to_segment_noloop(points)
    is_intersecting = False
    for i, segment in enumerate(segments):
        is_intersecting = is_circle_intersecting_with_line(circle_center, circle_radius, segment[0], segment[1]) or is_intersecting 
        # if is_circle_intersecting_with_line(circle_center, circle_radius, segment[0], segment[1]):
    return is_intersecting
    
def get_circle_intersection_with_points(circle_center, circle_radius, points):
    segments = convert_points_to_segment_noloop(points)
    is_intersecting = False
    index = None
    for i, segment in enumerate(segments):
        is_intersecting = is_circle_intersecting_with_line(circle_center, circle_radius, segment[0], segment[1]) or is_intersecting 
        if is_circle_intersecting_with_line(circle_center, circle_radius, segment[0], segment[1]):
            print(segment[0], segment[1])
            print("")
            print(circle_center)
            print("")
            print(points)
            index = i
    return is_intersecting, index

def normalize(v):
    """Returns the unit vector of v."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def get_boundary_for_point(
    prev_point, center_point, next_point, width
):
    """
    Computes the left and right boundary points for a given center point.

    Parameters:
    - prev_point: the point preceding the center point (None if center_point is the first point).
    - center_point: the point for which to compute the boundaries.
    - next_point: the point succeeding the center point (None if center_point is the last point).
    - width: the track width.

    Returns:
    - left_boundary: the left boundary point for the given center point.
    - right_boundary: the right boundary point for the given center point.
    """
    # Convert to numpy arrays for easier calculations
    if prev_point is not None:
        prev_point = np.array(prev_point)
    center_point = np.array(center_point)
    if next_point is not None:
        next_point = np.array(next_point)

    # Compute the direction
    if prev_point is None:  # First point
        direction = next_point - center_point
    elif next_point is None:  # Last point
        direction = center_point - prev_point
    else:  # Middle points
        direction = next_point - prev_point

    # Normalize the direction vector
    direction = normalize(direction)

    # Compute the normal vector (perpendicular)
    normal = np.array([-direction[1], direction[0]])

    # Compute left and right boundary points
    left = center_point + (width / 2) * normal
    right = center_point - (width / 2) * normal

    return list(left), list(right)