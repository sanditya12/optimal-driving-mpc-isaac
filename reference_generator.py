import numpy as np
from typing import List, Tuple


class ReferenceGenerator:
    def __init__(
        self, horizon: int, center_points: List[Tuple[float, float]]
    ) -> None:
        self.horizon = horizon
        self.center_points = center_points

    def convert3d2d(center_points_3d: List[Tuple[float,float]])-> Tuple[float,float]:
        transformed = [sublist[:2] for sublist in center_points_3d]
        return transformed


    def generate_map(
        self, position: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        index = self.get_closest_position_index(position, self.center_points)

        return self.wrap_slice(self.center_points, index, self.horizon)

    def wrap_slice(self, t: Tuple, start: int, length: int) -> Tuple:
        # Get the initial slice
        result = t[start : start + length]

        # Check if we've reached or exceeded the end of the tuple
        if len(result) < length:
            required = length - len(result)
            result += t[:required]
            # start = required

        return result

    def get_closest_position_index(
        self, pos: Tuple[float, float], positions: List[Tuple[float, float]]
    ) -> int:
        # Calculate the squared distance to avoid sqrt calculation for efficiency
        def squared_distance(
            p1: Tuple[float, float], p2: Tuple[float, float]
        ) -> float:
            return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

        # Find the index of the position with the minimum squared distance
        return min(
            enumerate(positions), key=lambda x: squared_distance(pos, x[1])
        )[0]
    
    def find_projection(self, position):
        min_distance = float('inf')
        closest_point = None
    
        for i in range(len(self.center_points) - 1):
            a = self.center_points[i]
            b = self.center_points[i + 1]
    
            projected_point = self.closest_point_on_segment(position, a, b)
    
            distance = ((position[0] - projected_point[0]) ** 2 + (position[1] - projected_point[1]) ** 2) ** 0.5
    
            if distance < min_distance:
                min_distance = distance
                closest_point = projected_point
        
        xc, yc = closest_point
        return np.array([xc,yc,0])
    
    def closest_point_on_segment(self, p, a, b):
        ap = [p[0] - a[0], p[1] - a[1]]
        ab = [b[0] - a[0], b[1] - a[1]]
    
        ab2 = self.dot(ab, ab)
        ap_ab = self.dot(ap, ab)
    
        t = min(1, max(0, ap_ab / ab2))
    
        return [a[0] + ab[0] * t, a[1] + ab[1] * t]
    
    def dot(self, a, b):
        return a[0] * b[0] + a[1] * b[1]
    
    def get_boundary_for_point(
        self, prev_point, center_point, next_point, width
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
        direction = self.normalize(direction)

        # Compute the normal vector (perpendicular)
        normal = np.array([-direction[1], direction[0]])

        # Compute left and right boundary points
        left = center_point + (width / 2) * normal
        right = center_point - (width / 2) * normal

        return list(left), list(right)

    def normalize(self, v):
        """Returns the unit vector of v."""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def is_intersecting(self, circle_center, circle_radius, point1, point2):
        """Check if a circle is intersecting a line defined by two points."""

        x0, y0 = circle_center
        x1, y1 = point1
        x2, y2 = point2

        # Calculate the perpendicular distance from the circle center to the line
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distance = numerator / denominator

        # Check if circle intersects the line
        return distance <= circle_radius
    
    def distance_from_point_to_line(self, px,py,x1,y1,x2,y2):
        if x2 - x1 == 0:
            return abs(px-x1)
        
        m = (y2-y1)/(x2-x1)
        c = y1 - m*x1
        b = -1

        numerator = abs(m*px + b*py + c)
        denominator = (m**2 + b**2)**0.5
        return numerator/denominator
    
    def get_max_min_coordinates(self, array):
        max_values = [max(col) for col in zip(*array)]
        min_values = [min(col) for col in zip(*array)]
        return max_values, min_values

    def get_normalized_value(self, value, max, min):
        return (value - min)/(max-min)

    def check_equal(self, arr1, arr2, threshold = 1e-3):
        if len(arr1) != len(arr2):
            return False
        
        for a,b in zip(arr1,arr2):
            if abs(a - b)> threshold:
                return False
        return True
