import abc
import typing
import dataclasses

import numpy as np

@dataclasses.dataclass
class RoadSegment(abc.ABC):
    @abc.abstractmethod
    def get_start_pose(self) -> typing.Tuple[np.ndarray, float]:
        """Return (position, heading) at start of segment."""
        pass

    @abc.abstractmethod
    def get_end_pose(self) -> typing.Tuple[np.ndarray, float]:
        """Return (position, heading) at end of segment."""
        pass

    @abc.abstractmethod
    def contains_point(self, point: np.ndarray, half_width: float) -> bool:
        """Check if point is within the road segment bounds."""
        pass

    @abc.abstractmethod
    def get_centerline_points(self, num_points: int) -> np.ndarray:
        """Get points along the centerline."""
        pass

    @abc.abstractmethod
    def get_boundary_points(
        self, half_width: float, num_points: int
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Get left and right boundary points."""
        pass

    @abc.abstractmethod
    def get_length(self) -> float:
        """Get the length of this segment along the centerline."""
        pass

    @abc.abstractmethod
    def get_bounds(self, half_width: float) -> typing.Tuple[float, float, float, float]:
        """Get (min_x, min_y, max_x, max_y) bounds of this segment."""
        pass


@dataclasses.dataclass
class StraightSegment(RoadSegment):
    start: typing.Tuple[float, float]
    heading: float  # radians
    length: float   # meters

    def _start_array(self) -> np.ndarray:
        return np.array(self.start, dtype=np.float32)

    def _direction(self) -> np.ndarray:
        return np.array([np.cos(self.heading), np.sin(self.heading)], dtype=np.float32)

    def _perpendicular(self) -> np.ndarray:
        return np.array([-np.sin(self.heading), np.cos(self.heading)], dtype=np.float32)

    def get_start_pose(self) -> typing.Tuple[np.ndarray, float]:
        return self._start_array(), self.heading

    def get_end_pose(self) -> typing.Tuple[np.ndarray, float]:
        end_pos = self._start_array() + self._direction() * self.length
        return end_pos, self.heading

    def contains_point(self, point: np.ndarray, half_width: float) -> bool:
        start = self._start_array()
        direction = self._direction()
        to_point = point - start
        along = np.dot(to_point, direction)
        across = np.dot(to_point, self._perpendicular())
        return 0 <= along <= self.length and abs(across) <= half_width

    def get_centerline_points(self, num_points: int) -> np.ndarray:
        start = self._start_array()
        direction = self._direction()
        t = np.linspace(0, self.length, num_points)
        return start + np.outer(t, direction)

    def get_boundary_points(
        self, half_width: float, num_points: int
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        centerline = self.get_centerline_points(num_points)
        perp = self._perpendicular()
        left = centerline + perp * half_width
        right = centerline - perp * half_width
        return left, right

    def get_length(self) -> float:
        return self.length

    def get_bounds(self, half_width: float) -> typing.Tuple[float, float, float, float]:
        left, right = self.get_boundary_points(half_width, 2)
        all_points = np.vstack([left, right])
        return (
            float(np.min(all_points[:, 0])),
            float(np.min(all_points[:, 1])),
            float(np.max(all_points[:, 0])),
            float(np.max(all_points[:, 1])),
        )

@dataclasses.dataclass
class ArcSegment(RoadSegment):
    """
    A curved road segment (arc of a circle).

    Positive arc_angle turns left (counter-clockwise), negative turns right (clockwise).
    """

    start: typing.Tuple[float, float]
    start_heading: float  # radians
    radius: float         # meters, always positive
    arc_angle: float      # radians, positive = left, negative = right

    def _start_array(self) -> np.ndarray:
        return np.array(self.start, dtype=np.float32)

    def _get_center(self) -> np.ndarray:
        """Compute the center of the arc circle."""
        start = self._start_array()
        if self.arc_angle >= 0:
            perp = np.array(
                [-np.sin(self.start_heading), np.cos(self.start_heading)],
                dtype=np.float32,
            )
        else:
            perp = np.array(
                [np.sin(self.start_heading), -np.cos(self.start_heading)],
                dtype=np.float32,
            )
        return start + perp * self.radius

    def _get_start_angle(self) -> float:
        """Get the angle from center to start point."""
        center = self._get_center()
        start = self._start_array()
        diff = start - center
        return float(np.arctan2(diff[1], diff[0]))

    def get_start_pose(self) -> typing.Tuple[np.ndarray, float]:
        return self._start_array(), self.start_heading

    def get_end_pose(self) -> typing.Tuple[np.ndarray, float]:
        center = self._get_center()
        start_angle = self._get_start_angle()
        end_angle = start_angle + self.arc_angle

        end_pos = center + self.radius * np.array(
            [np.cos(end_angle), np.sin(end_angle)], dtype=np.float32
        )
        end_heading = self.start_heading + self.arc_angle
        end_heading = float(np.arctan2(np.sin(end_heading), np.cos(end_heading)))
        return end_pos, end_heading

    def contains_point(self, point: np.ndarray, half_width: float) -> bool:
        center = self._get_center()
        dist_to_center = float(np.linalg.norm(point - center))

        # Check radial distance
        inner_radius = self.radius - half_width
        outer_radius = self.radius + half_width
        if not (inner_radius <= dist_to_center <= outer_radius):
            return False

        # Check angular position
        start_angle = self._get_start_angle()
        point_angle = float(np.arctan2(point[1] - center[1], point[0] - center[0]))

        if self.arc_angle >= 0:
            angle_diff = point_angle - start_angle
            while angle_diff < 0:
                angle_diff += 2 * np.pi
            while angle_diff > 2 * np.pi:
                angle_diff -= 2 * np.pi
            return 0 <= angle_diff <= self.arc_angle
        else:
            angle_diff = start_angle - point_angle
            while angle_diff < 0:
                angle_diff += 2 * np.pi
            while angle_diff > 2 * np.pi:
                angle_diff -= 2 * np.pi
            return 0 <= angle_diff <= abs(self.arc_angle)

    def get_centerline_points(self, num_points: int) -> np.ndarray:
        center = self._get_center()
        start_angle = self._get_start_angle()
        angles = np.linspace(start_angle, start_angle + self.arc_angle, num_points)
        points = np.zeros((num_points, 2), dtype=np.float32)
        points[:, 0] = center[0] + self.radius * np.cos(angles)
        points[:, 1] = center[1] + self.radius * np.sin(angles)
        return points

    def get_boundary_points(
        self, half_width: float, num_points: int
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        center = self._get_center()
        start_angle = self._get_start_angle()
        angles = np.linspace(start_angle, start_angle + self.arc_angle, num_points)

        inner_radius = self.radius - half_width
        outer_radius = self.radius + half_width

        inner = np.zeros((num_points, 2), dtype=np.float32)
        outer = np.zeros((num_points, 2), dtype=np.float32)

        inner[:, 0] = center[0] + inner_radius * np.cos(angles)
        inner[:, 1] = center[1] + inner_radius * np.sin(angles)
        outer[:, 0] = center[0] + outer_radius * np.cos(angles)
        outer[:, 1] = center[1] + outer_radius * np.sin(angles)

        if self.arc_angle >= 0:
            return outer, inner
        else:
            return inner, outer

    def get_length(self) -> float:
        return abs(self.arc_angle) * self.radius

    def get_bounds(self, half_width: float) -> typing.Tuple[float, float, float, float]:
        num_points = max(20, int(self.get_length()))
        left, right = self.get_boundary_points(half_width, num_points)
        all_points = np.vstack([left, right])
        return (
            float(np.min(all_points[:, 0])),
            float(np.min(all_points[:, 1])),
            float(np.max(all_points[:, 0])),
            float(np.max(all_points[:, 1])),
        )


@dataclasses.dataclass
class Road:
    """
    A single-lane road composed of connected segments.
    """

    segments: typing.List[RoadSegment]
    width: float = 4.0

    @property
    def half_width(self) -> float:
        return self.width / 2

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is on this road."""
        for segment in self.segments:
            if segment.contains_point(point, self.half_width):
                return True
        return False

    def get_total_length(self) -> float:
        """Get total road length."""
        return sum(seg.get_length() for seg in self.segments)

    def get_bounds(self) -> typing.Tuple[float, float, float, float]:
        """Get (min_x, min_y, max_x, max_y) bounds of this road."""
        if not self.segments:
            return (0, 0, 0, 0)

        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        for segment in self.segments:
            bounds = segment.get_bounds(self.half_width)
            min_x = min(min_x, bounds[0])
            min_y = min(min_y, bounds[1])
            max_x = max(max_x, bounds[2])
            max_y = max(max_y, bounds[3])

        return (min_x, min_y, max_x, max_y)

    @property
    def start_pose(self) -> typing.Tuple[np.ndarray, float]:
        """Get the starting pose of the road."""
        if not self.segments:
            return np.zeros(2, dtype=np.float32), 0.0
        return self.segments[0].get_start_pose()

    @property
    def end_pose(self) -> typing.Tuple[np.ndarray, float]:
        """Get the ending pose of the road."""
        if not self.segments:
            return np.zeros(2, dtype=np.float32), 0.0
        return self.segments[-1].get_end_pose()


class RoadBuilder:
    def __init__(self, start: typing.Tuple[float, float], heading: float, width: float = 4.0):
        """
        Args:
            start: Starting position (x, y) in meters
            heading: Starting heading in radians
            width: Road width in meters
        """
        self._current_pos = np.array(start, dtype=np.float32)
        self._current_heading = heading
        self._segments: typing.List[RoadSegment] = []
        self._width = width

    def straight(self, length: float) -> "RoadBuilder":
        segment = StraightSegment(
            start=tuple(self._current_pos),
            heading=self._current_heading,
            length=length,
        )
        self._segments.append(segment)
        self._current_pos, self._current_heading = segment.get_end_pose()
        return self

    def turn_left(self, radius: float, angle_degrees: float) -> "RoadBuilder":
        segment = ArcSegment(
            start=tuple(self._current_pos),
            start_heading=self._current_heading,
            radius=radius,
            arc_angle=np.radians(angle_degrees),
        )
        self._segments.append(segment)
        self._current_pos, self._current_heading = segment.get_end_pose()
        return self

    def turn_right(self, radius: float, angle_degrees: float) -> "RoadBuilder":
        segment = ArcSegment(
            start=tuple(self._current_pos),
            start_heading=self._current_heading,
            radius=radius,
            arc_angle=-np.radians(angle_degrees),
        )
        self._segments.append(segment)
        self._current_pos, self._current_heading = segment.get_end_pose()
        return self

    def turn(self, radius: float, angle_degrees: float) -> "RoadBuilder":
        """Add a turn. Positive angle = left, negative = right."""
        if angle_degrees >= 0:
            return self.turn_left(radius, angle_degrees)
        else:
            return self.turn_right(radius, abs(angle_degrees))

    def build(self) -> Road:
        return Road(segments=self._segments, width=self._width)

@dataclasses.dataclass
class RoadNetwork:
    """A collection of roads forming a network."""

    roads: typing.List[Road] = dataclasses.field(default_factory=list)

    def add_road(self, road: Road) -> None:
        self.roads.append(road)

    def contains_point(self, point: np.ndarray) -> bool:
        for road in self.roads:
            if road.contains_point(point):
                return True
        return False

    def is_off_road(self, point: np.ndarray) -> bool:
        return not self.contains_point(point)

    def get_bounds(self) -> typing.Tuple[float, float, float, float]:
        """Get (min_x, min_y, max_x, max_y) bounds of the entire network."""
        if not self.roads:
            return (0, 0, 0, 0)

        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        for road in self.roads:
            bounds = road.get_bounds()
            min_x = min(min_x, bounds[0])
            min_y = min(min_y, bounds[1])
            max_x = max(max_x, bounds[2])
            max_y = max(max_y, bounds[3])

        return (min_x, min_y, max_x, max_y)

def create_oval_track(
    center: typing.Tuple[float, float],
    straight_length: float,
    turn_radius: float,
    width: float = 4.0,
) -> Road:
    start_x = center[0] - straight_length / 2
    start_y = center[1] - turn_radius
    return (
        RoadBuilder((start_x, start_y), 0.0, width)
        .straight(straight_length)
        .turn_left(turn_radius, 180)
        .straight(straight_length)
        .turn_left(turn_radius, 180)
        .build()
    )

def create_rectangular_track(
    center: typing.Tuple[float, float],
    length: float,
    height: float,
    turn_radius: float,
    width: float = 4.0,
) -> Road:
    """
    Create a rectangular track (closed loop with rounded corners).

    Args:
        center: Center of the rectangle (x, y) in meters
        length: Outer length (horizontal) of the rectangle in meters
        height: Outer height (vertical) of the rectangle in meters
        turn_radius: Radius of the corner turns in meters
        width: Road width in meters

    Returns:
        A Road forming a rectangular loop
    """
    straight_long = length - 2 * turn_radius
    straight_short = height - 2 * turn_radius

    if straight_long <= 0 or straight_short <= 0:
        raise ValueError(
            f"Turn radius {turn_radius}m is too large for rectangle "
            f"{length}m x {height}m. Max turn radius: {min(length, height) / 2}m"
        )

    # Start at bottom-left corner, pointing right
    cx, cy = center
    start_x = cx - length / 2 + turn_radius
    start_y = cy - height / 2

    return (
        RoadBuilder((start_x, start_y), 0.0, width)
        # Bottom edge (going right)
        .straight(straight_long)
        .turn_left(turn_radius, 90)
        # Right edge (going up)
        .straight(straight_short)
        .turn_left(turn_radius, 90)
        # Top edge (going left)
        .straight(straight_long)
        .turn_left(turn_radius, 90)
        # Left edge (going down)
        .straight(straight_short)
        .turn_left(turn_radius, 90)
        .build()
    )
