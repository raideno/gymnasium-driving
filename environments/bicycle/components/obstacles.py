import abc
import typing
import dataclasses

import numpy as np

MotionFn = typing.Callable[[float, typing.Tuple[float, float]], typing.Tuple[float, float]]

def static() -> MotionFn:
    return lambda t, init: init

def circular(radius: float, angular_speed: float, center: typing.Tuple[float, float] = None) -> MotionFn:
    def motion(t: float, init: typing.Tuple[float, float]) -> typing.Tuple[float, float]:
        cx, cy = center if center is not None else (init[0], init[1] - radius)
        angle = angular_speed * t
        return (cx + radius * np.cos(angle), cy + radius * np.sin(angle))
    return motion

def linear(velocity: typing.Tuple[float, float]) -> MotionFn:
    def motion(t: float, init: typing.Tuple[float, float]) -> typing.Tuple[float, float]:
        return (init[0] + velocity[0] * t, init[1] + velocity[1] * t)
    return motion

def oscillate(axis: str, amplitude: float, frequency: float) -> MotionFn:
    def motion(t: float, init: typing.Tuple[float, float]) -> typing.Tuple[float, float]:
        offset = amplitude * np.sin(2 * np.pi * frequency * t)
        if axis == 'x':
            return (init[0] + offset, init[1])
        else:
            return (init[0], init[1] + offset)
    return motion

def figure_eight(size: float, speed: float) -> MotionFn:
    def motion(t: float, init: typing.Tuple[float, float]) -> typing.Tuple[float, float]:
        angle = speed * t
        x = size * np.sin(angle)
        y = size * np.sin(angle) * np.cos(angle)
        return (init[0] + x, init[1] + y)
    return motion

def waypoints(
    points: typing.List[typing.Tuple[float, float]], 
    speed: float, 
    loop: bool = True
) -> MotionFn:
    """
    Move along a series of waypoints at a constant speed.
    """
    def motion(t: float, init: typing.Tuple[float, float]) -> typing.Tuple[float, float]:
        if len(points) == 0:
            return init
        if len(points) == 1:
            return points[0]
        
        # Calculate total path length and segment lengths
        segments = []
        total_length = 0.0
        for i in range(len(points)):
            if i == len(points) - 1:
                if loop:
                    next_pt = points[0]
                else:
                    break
            else:
                next_pt = points[i + 1]
            seg_len = np.sqrt((next_pt[0] - points[i][0])**2 + (next_pt[1] - points[i][1])**2)
            segments.append((points[i], next_pt, seg_len))
            total_length += seg_len
        
        if total_length == 0:
            return points[0]
        
        # Find current position along path
        distance = (speed * t) % total_length if loop else min(speed * t, total_length)
        
        accumulated = 0.0
        for start_pt, end_pt, seg_len in segments:
            if accumulated + seg_len >= distance:
                # Interpolate within this segment
                ratio = (distance - accumulated) / seg_len if seg_len > 0 else 0
                x = start_pt[0] + ratio * (end_pt[0] - start_pt[0])
                y = start_pt[1] + ratio * (end_pt[1] - start_pt[1])
                return (x, y)
            accumulated += seg_len
        
        return points[-1] if not loop else points[0]
    return motion

@dataclasses.dataclass
class Obstacle(abc.ABC):
    center: typing.Tuple[float, float]
    motion: MotionFn = dataclasses.field(default_factory=static)
    
    def __post_init__(self):
        self._initial_center = self.center
        self._current_time = 0.0

    def update(self, time: float) -> None:
        self._current_time = time
        self.center = self.motion(time, self._initial_center)
    
    def reset(self) -> None:
        self._current_time = 0.0
        self.center = self._initial_center

    @abc.abstractmethod
    def check_collision(self, point: np.ndarray) -> bool:
        pass

    @abc.abstractmethod
    def get_type(self) -> str:
        pass

    @abc.abstractmethod
    def get_bounds(self) -> typing.Tuple[float, float, float, float]:
        pass


@dataclasses.dataclass
class Circle(Obstacle):
    radius: float = 0.0

    def check_collision(self, point: np.ndarray) -> bool:
        center_array = np.array(self.center, dtype=np.float32)
        dist = np.linalg.norm(point - center_array)
        return dist <= self.radius

    def get_type(self) -> str:
        return "circle"

    def get_bounds(self) -> typing.Tuple[float, float, float, float]:
        return (
            self.center[0] - self.radius,
            self.center[1] - self.radius,
            self.center[0] + self.radius,
            self.center[1] + self.radius,
        )

@dataclasses.dataclass
class Rectangle(Obstacle):
    width: float = 0.0
    height: float = 0.0

    def check_collision(self, point: np.ndarray) -> bool:
        center_array = np.array(self.center, dtype=np.float32)
        half_width = self.width / 2
        half_height = self.height / 2

        dx = abs(point[0] - center_array[0])
        dy = abs(point[1] - center_array[1])

        return dx <= half_width and dy <= half_height

    def get_type(self) -> str:
        return "rectangle"

    def get_bounds(self) -> typing.Tuple[float, float, float, float]:
        return (
            self.center[0] - self.width / 2,
            self.center[1] - self.height / 2,
            self.center[0] + self.width / 2,
            self.center[1] + self.height / 2,
        )
