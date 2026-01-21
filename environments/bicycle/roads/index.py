# roads/index.py

import abc
import enum
import typing
import dataclasses

import numpy as np

class LaneDirection(enum.Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    BIDIRECTIONAL = "bidirectional"

@dataclasses.dataclass
class LaneConfig:
    num_lanes: int = 2
    lane_width: float = 3.5
    direction: LaneDirection = LaneDirection.BIDIRECTIONAL

    @property
    def total_width(self) -> float:
        return self.num_lanes * self.lane_width

    @property
    def half_width(self) -> float:
        return self.total_width / 2

SINGLE_LANE = LaneConfig(num_lanes=1, lane_width=3.5)
DOUBLE_LANE = LaneConfig(num_lanes=2, lane_width=3.5)
TRIPLE_LANE = LaneConfig(num_lanes=3, lane_width=3.5)
HIGHWAY_LANES = LaneConfig(num_lanes=4, lane_width=3.7)

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
        """Get points along the centerline for rendering."""
        pass

    @abc.abstractmethod
    def get_boundary_points(
        self, half_width: float, num_points: int
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Get left and right boundary points for rendering."""
        pass

    @abc.abstractmethod
    def get_length(self) -> float:
        """Get the length of this segment along the centerline."""
        pass

    @abc.abstractmethod
    def get_bounds(self, half_width: float) -> typing.Tuple[float, float, float, float]:
        """Get (min_x, min_y, max_x, max_y) bounds of this segment."""
        pass

    @abc.abstractmethod
    def get_position_info(
        self, point: np.ndarray, half_width: float
    ) -> typing.Dict[str, float]:
        """
        Get detailed position information for a point on/near the segment.

        Returns dict with:
            - 'along': distance along centerline
            - 'across': perpendicular distance from centerline
            - 'is_on_segment': 1.0 if point is on segment, 0.0 otherwise
        """
        pass


@dataclasses.dataclass
class StraightSegment(RoadSegment):
    start: typing.Tuple[float, float]
    heading: float
    length: float

    def _start_array(self) -> np.ndarray:
        return np.array(self.start, dtype=np.float32)

    def _direction(self) -> np.ndarray:
        return np.array(
            [np.cos(self.heading), np.sin(self.heading)], dtype=np.float32
        )

    def _perpendicular(self) -> np.ndarray:
        return np.array(
            [-np.sin(self.heading), np.cos(self.heading)], dtype=np.float32
        )

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

    def get_bounds(
        self, half_width: float
    ) -> typing.Tuple[float, float, float, float]:
        left, right = self.get_boundary_points(half_width, 2)
        all_points = np.vstack([left, right])
        return (
            float(np.min(all_points[:, 0])),
            float(np.min(all_points[:, 1])),
            float(np.max(all_points[:, 0])),
            float(np.max(all_points[:, 1])),
        )

    def get_position_info(
        self, point: np.ndarray, half_width: float
    ) -> typing.Dict[str, float]:
        start = self._start_array()
        direction = self._direction()
        perp = self._perpendicular()

        to_point = point - start
        along = float(np.dot(to_point, direction))
        across = float(np.dot(to_point, perp))

        is_on_segment = float(
            0 <= along <= self.length and abs(across) <= half_width
        )

        return {
            "along": along,
            "across": across,
            "is_on_segment": is_on_segment,
        }

@dataclasses.dataclass
class ArcSegment(RoadSegment):
    """
    A curved road segment (arc of a circle).

    The arc starts at `start` with `start_heading` and curves with the given
    radius. Positive `arc_angle` turns left (counter-clockwise), negative
    turns right (clockwise).
    """

    start: typing.Tuple[float, float]
    start_heading: float  # radians
    radius: float  # meters, always positive
    arc_angle: float  # radians, positive = left turn, negative = right turn

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

        inner_radius = self.radius - half_width
        outer_radius = self.radius + half_width
        if not (inner_radius <= dist_to_center <= outer_radius):
            return False

        start_angle = self._get_start_angle()
        point_angle = float(
            np.arctan2(point[1] - center[1], point[0] - center[0])
        )

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

    def get_bounds(
        self, half_width: float
    ) -> typing.Tuple[float, float, float, float]:
        num_points = max(20, int(self.get_length()))
        left, right = self.get_boundary_points(half_width, num_points)
        all_points = np.vstack([left, right])
        return (
            float(np.min(all_points[:, 0])),
            float(np.min(all_points[:, 1])),
            float(np.max(all_points[:, 0])),
            float(np.max(all_points[:, 1])),
        )

    def get_position_info(
        self, point: np.ndarray, half_width: float
    ) -> typing.Dict[str, float]:
        center = self._get_center()
        dist_to_center = float(np.linalg.norm(point - center))

        inner_radius = self.radius - half_width
        outer_radius = self.radius + half_width
        on_radial = (
            1.0 if inner_radius <= dist_to_center <= outer_radius else 0.0
        )

        start_angle = self._get_start_angle()
        point_angle = float(
            np.arctan2(point[1] - center[1], point[0] - center[0])
        )
        across = dist_to_center - self.radius

        if self.arc_angle >= 0:
            angle_diff = point_angle - start_angle
            while angle_diff < 0:
                angle_diff += 2 * np.pi
            while angle_diff > 2 * np.pi:
                angle_diff -= 2 * np.pi
            on_angular = float(0 <= angle_diff <= self.arc_angle)
            along = angle_diff * self.radius
        else:
            angle_diff = start_angle - point_angle
            while angle_diff < 0:
                angle_diff += 2 * np.pi
            while angle_diff > 2 * np.pi:
                angle_diff -= 2 * np.pi
            on_angular = float(0 <= angle_diff <= abs(self.arc_angle))
            along = angle_diff * self.radius

        is_on_segment = float(on_radial > 0.5 and on_angular > 0.5)

        return {
            "along": along,
            "across": across,
            "is_on_segment": is_on_segment,
        }


@dataclasses.dataclass
class Road:
    """
    A road composed of connected segments with lane configuration.

    Segments are automatically connected end-to-end.
    """

    segments: typing.List[RoadSegment]
    lane_config: LaneConfig = dataclasses.field(default_factory=lambda: DOUBLE_LANE)
    # Rendering hints for junction handling
    # Specifies how much to trim from boundaries at each end (in meters)
    # This prevents boundary lines from extending into intersecting roads
    start_trim: float = 0.0
    end_trim: float = 0.0

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is on this road."""
        half_width = self.lane_config.half_width
        for segment in self.segments:
            if segment.contains_point(point, half_width):
                return True
        return False

    def get_total_length(self) -> float:
        """Get total road length."""
        return sum(seg.get_length() for seg in self.segments)

    def get_bounds(self) -> typing.Tuple[float, float, float, float]:
        """Get (min_x, min_y, max_x, max_y) bounds of this road."""
        if not self.segments:
            return (0, 0, 0, 0)

        half_width = self.lane_config.half_width
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        for segment in self.segments:
            bounds = segment.get_bounds(half_width)
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

    def get_road_info(self, point: np.ndarray) -> typing.Dict[str, typing.Any]:
        """Get comprehensive road information for a point."""
        half_width = self.lane_config.half_width

        on_road = False
        min_dist_to_center = float("inf")
        distance_to_left = float("inf")
        distance_to_right = float("inf")
        along_segment = 0.0
        segment_index = -1

        for idx, segment in enumerate(self.segments):
            pos_info = segment.get_position_info(point, half_width)
            dist_to_center = abs(pos_info["across"])

            if pos_info["is_on_segment"] > 0.5:
                on_road = True
                segment_index = idx
                along_segment = pos_info["along"]
                min_dist_to_center = 0.0
                across = pos_info["across"]
                distance_to_left = half_width - across
                distance_to_right = half_width + across
                break
            elif dist_to_center < min_dist_to_center:
                min_dist_to_center = dist_to_center
                along_segment = pos_info["along"]
                segment_index = idx
                across = pos_info["across"]
                distance_to_left = half_width - across
                distance_to_right = half_width + across

        return {
            "on_road": on_road,
            "distance_to_center": (
                min_dist_to_center
                if min_dist_to_center != float("inf")
                else float("inf")
            ),
            "distance_to_left_boundary": max(0.0, distance_to_left),
            "distance_to_right_boundary": max(0.0, distance_to_right),
            "along_segment": along_segment,
            "segment_index": segment_index,
        }


class RoadBuilder:
    """Builder for creating roads with chained segments."""

    def __init__(
        self,
        start: typing.Tuple[float, float],
        heading: float,
        lane_config: LaneConfig = DOUBLE_LANE,
    ):
        self._current_pos = np.array(start, dtype=np.float32)
        self._current_heading = heading
        self._segments: typing.List[RoadSegment] = []
        self._lane_config = lane_config

    def straight(self, length: float) -> "RoadBuilder":
        """Add a straight segment (length in meters)."""
        segment = StraightSegment(
            start=tuple(self._current_pos),
            heading=self._current_heading,
            length=length,
        )
        self._segments.append(segment)
        end_pos, end_heading = segment.get_end_pose()
        self._current_pos = end_pos
        self._current_heading = end_heading
        return self

    def turn_left(self, radius: float, angle_degrees: float) -> "RoadBuilder":
        """Add a left turn (radius in meters)."""
        arc_angle = np.radians(angle_degrees)
        segment = ArcSegment(
            start=tuple(self._current_pos),
            start_heading=self._current_heading,
            radius=radius,
            arc_angle=arc_angle,
        )
        self._segments.append(segment)
        end_pos, end_heading = segment.get_end_pose()
        self._current_pos = end_pos
        self._current_heading = end_heading
        return self

    def turn_right(self, radius: float, angle_degrees: float) -> "RoadBuilder":
        """Add a right turn (radius in meters)."""
        arc_angle = -np.radians(angle_degrees)
        segment = ArcSegment(
            start=tuple(self._current_pos),
            start_heading=self._current_heading,
            radius=radius,
            arc_angle=arc_angle,
        )
        self._segments.append(segment)
        end_pos, end_heading = segment.get_end_pose()
        self._current_pos = end_pos
        self._current_heading = end_heading
        return self

    def turn(self, radius: float, angle_degrees: float) -> "RoadBuilder":
        """Add a turn. Positive angle = left, negative = right."""
        if angle_degrees >= 0:
            return self.turn_left(radius, angle_degrees)
        else:
            return self.turn_right(radius, abs(angle_degrees))

    def build(self) -> Road:
        """Build and return the road."""
        return Road(segments=self._segments, lane_config=self._lane_config)


@dataclasses.dataclass
class RoadNetwork:
    """A collection of roads forming a network."""

    roads: typing.List[Road] = dataclasses.field(default_factory=list)

    def add_road(self, road: Road) -> None:
        """Add a road to the network."""
        self.roads.append(road)

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is on any road in the network."""
        for road in self.roads:
            if road.contains_point(point):
                return True
        return False

    def is_off_road(self, point: np.ndarray) -> bool:
        """Check if a point is off all roads."""
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

    def get_comprehensive_road_info(
        self, point: np.ndarray, heading: float
    ) -> typing.Dict[str, typing.Any]:
        """Get comprehensive road and lane information for vehicle position."""
        best_road = None
        best_road_info = None
        min_center_dist = float("inf")

        for road in self.roads:
            road_info = road.get_road_info(point)
            dist_to_center = road_info["distance_to_center"]

            if road_info["on_road"]:
                best_road = road
                best_road_info = road_info
                break
            elif dist_to_center < min_center_dist:
                best_road = road
                best_road_info = road_info
                min_center_dist = dist_to_center

        if best_road_info is None or not best_road:
            return {
                "on_road": False,
                "distance_to_road_center": float("inf"),
                "distance_to_left_boundary": float("inf"),
                "distance_to_right_boundary": float("inf"),
                "lane_number": -1,
                "num_lanes": 0,
                "lane_width": 0.0,
                "road_width": 0.0,
                "distance_to_lane_center": float("inf"),
                "intersecting_left_boundary": False,
                "intersecting_right_boundary": False,
                "in_single_lane": False,
                "in_multiple_lanes": False,
                "relative_heading": 0.0,
            }

        on_road = best_road_info["on_road"]
        dist_left = best_road_info["distance_to_left_boundary"]
        dist_right = best_road_info["distance_to_right_boundary"]
        lane_width = best_road.lane_config.lane_width
        num_lanes = best_road.lane_config.num_lanes
        road_width = best_road.lane_config.total_width

        across_dist = dist_left - best_road.lane_config.half_width
        lane_number = int((num_lanes - 1) / 2) - int(
            across_dist / lane_width
        )
        lane_number = max(0, min(lane_number, num_lanes - 1))

        # Compute distance to current lane center
        # Lane centers are positioned at regular intervals from the road center
        # For num_lanes=1: lane 0 center is at road center (offset=0)
        # For num_lanes=2: lane 0 at -lane_width/2, lane 1 at +lane_width/2
        # General formula: offset from road center for lane i
        lane_center_offset = (lane_number - (num_lanes - 1) / 2) * lane_width
        distance_to_lane_center = best_road_info["distance_to_center"] - lane_center_offset

        # Car half-width ~1m for boundary intersection check
        car_half_width = 1.0
        intersecting_left = dist_left < car_half_width
        intersecting_right = dist_right < car_half_width

        in_single_lane = not intersecting_left and not intersecting_right
        in_multiple_lanes = intersecting_left or intersecting_right

        relative_heading = 0.0

        return {
            "on_road": on_road,
            "distance_to_road_center": best_road_info["distance_to_center"],
            "distance_to_left_boundary": dist_left,
            "distance_to_right_boundary": dist_right,
            "lane_number": lane_number if on_road else -1,
            "num_lanes": num_lanes,
            "lane_width": lane_width,
            "road_width": road_width,
            "distance_to_lane_center": distance_to_lane_center,
            "intersecting_left_boundary": intersecting_left,
            "intersecting_right_boundary": intersecting_right,
            "in_single_lane": in_single_lane,
            "in_multiple_lanes": in_multiple_lanes,
            "relative_heading": relative_heading,
        }
