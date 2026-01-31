# roads/predefined.py

import typing

import numpy as np

from .index import *

def create_oval_track(
    center: typing.Tuple[float, float],
    straight_length: float,
    turn_radius: float,
    lane_config: LaneConfig = DOUBLE_LANE,
) -> Road:
    """Create an oval/racetrack shaped road."""
    start_x = center[0] - straight_length / 2
    start_y = center[1] - turn_radius
    return (
        RoadBuilder((start_x, start_y), 0.0, lane_config)
        .straight(straight_length)
        .turn_left(turn_radius, 180)
        .straight(straight_length)
        .turn_left(turn_radius, 180)
        .build()
    )

def create_figure_eight(
    center: typing.Tuple[float, float],
    radius: float,
    lane_config: LaneConfig = DOUBLE_LANE,
) -> Road:
    """Create a figure-8 shaped road."""
    start_x = center[0]
    start_y = center[1]
    return (
        RoadBuilder((start_x, start_y), np.pi / 2, lane_config)
        .turn_left(radius, 360)
        .turn_right(radius, 360)
        .build()
    )

# =============================================================================
# Enhanced Lane Configuration
# =============================================================================

def lane_config_from_width(
    total_width: float, num_lanes: int = 2
) -> LaneConfig:
    """
    Create a LaneConfig from a desired total road width.

    Args:
        total_width: Total road width in meters
        num_lanes: Number of lanes (for lane markings)

    Returns:
        LaneConfig with the specified total width
    """
    lane_width = total_width / num_lanes
    return LaneConfig(num_lanes=num_lanes, lane_width=lane_width)

def create_rectangular_track(
    center: typing.Tuple[float, float],
    length: float,
    width: float,
    turn_radius: float,
    lane_config: LaneConfig = DOUBLE_LANE,
    start_corner: str = "bottom_left",
) -> Road:
    """
    Create a rectangular track (closed loop).

    The track has 4 straight segments connected by 90-degree turns.

    Args:
        center: Center of the rectangle (x, y) in meters
        length: Length of the long sides in meters
        width: Length of the short sides in meters
        turn_radius: Radius of the corner turns in meters
        lane_config: Lane configuration for the road
        start_corner: Which corner to start from
            ("bottom_left", "bottom_right", "top_right", "top_left")

    Returns:
        A Road forming a rectangular loop
    """
    # Adjust straight lengths to account for turn radii
    # The turns eat into the rectangle dimensions
    straight_long = length - 2 * turn_radius
    straight_short = width - 2 * turn_radius

    if straight_long <= 0 or straight_short <= 0:
        raise ValueError(
            f"Turn radius {turn_radius}m is too large for rectangle "
            f"{length}m x {width}m. Max turn radius: "
            f"{min(length, width) / 2}m"
        )

    # Calculate starting position based on corner
    half_length = length / 2
    half_width = width / 2
    cx, cy = center

    if start_corner == "bottom_left":
        start_x = cx - half_length + turn_radius
        start_y = cy - half_width
        start_heading = 0.0  # pointing right
    elif start_corner == "bottom_right":
        start_x = cx + half_length
        start_y = cy - half_width + turn_radius
        start_heading = np.pi / 2  # pointing up
    elif start_corner == "top_right":
        start_x = cx + half_length - turn_radius
        start_y = cy + half_width
        start_heading = np.pi  # pointing left
    elif start_corner == "top_left":
        start_x = cx - half_length
        start_y = cy + half_width - turn_radius
        start_heading = -np.pi / 2  # pointing down
    else:
        raise ValueError(f"Unknown start_corner: {start_corner}")

    # Build the rectangular track: straight -> turn -> straight -> turn -> ...
    builder = RoadBuilder((start_x, start_y), start_heading, lane_config)

    # Go around the rectangle (4 sides, 4 turns)
    for i in range(4):
        if i % 2 == 0:
            builder.straight(straight_long)
        else:
            builder.straight(straight_short)
        builder.turn_left(turn_radius, 90)

    return builder.build()

def create_rectangular_track_with_cross(
    center: typing.Tuple[float, float],
    length: float,
    width: float,
    turn_radius: float,
    cross_width: float = 3.5,
    lane_config: LaneConfig = DOUBLE_LANE,
) -> RoadNetwork:
    """
    Create a rectangular track with a cross road dividing it in the middle.

    The outer rectangle forms the main track, and a cross-shaped road
    runs horizontally and vertically through the center, creating an
    intersection pattern.

    Args:
        center: Center of the rectangle (x, y) in meters
        length: Length of the long sides (horizontal) in meters
        width: Length of the short sides (vertical) in meters
        turn_radius: Radius of the corner turns in meters
        cross_width: Width of the cross roads in meters (for lane config)
        lane_config: Lane configuration for the main track
        
    Returns:
        A RoadNetwork containing the main track and the cross road
    """
    # Create main rectangular track
    main_track = create_rectangular_track(
        center=center,
        length=length,
        width=width,
        turn_radius=turn_radius,
        lane_config=lane_config,
    )

    # Create cross road lane config (narrower roads for crossing)
    cross_lane_config = lane_config_from_width(cross_width, num_lanes=1)

    half_length = length / 2
    half_width = width / 2
    cx, cy = center

    # Horizontal cross road (running left-right through center)
    horizontal_road = (
        RoadBuilder(
            (cx - half_length, cy),  # Start from left edge at center height
            0.0,  # Pointing right
            cross_lane_config,
        )
        .straight(length)
        .build()
    )

    # Vertical cross road (running top-bottom through center)
    vertical_road = (
        RoadBuilder(
            (cx, cy - half_width),  # Start from bottom edge at center width
            np.pi / 2,  # Pointing up
            cross_lane_config,
        )
        .straight(width)
        .build()
    )

    # Create network with all roads
    network = RoadNetwork()
    network.add_road(main_track)
    network.add_road(horizontal_road)
    network.add_road(vertical_road)

    return network


def create_square_track(
    center: typing.Tuple[float, float],
    side_length: float,
    turn_radius: float,
    lane_config: LaneConfig = DOUBLE_LANE,
) -> Road:
    """
    Create a square track (special case of rectangular track).

    Args:
        center: Center of the square (x, y) in meters
        side_length: Length of each side in meters
        turn_radius: Radius of the corner turns in meters
        lane_config: Lane configuration for the road

    Returns:
        A Road forming a square loop
    """
    return create_rectangular_track(
        center=center,
        length=side_length,
        width=side_length,
        turn_radius=turn_radius,
        lane_config=lane_config,
    )


def create_rounded_rectangle_track(
    center: typing.Tuple[float, float],
    length: float,
    width: float,
    lane_config: LaneConfig = DOUBLE_LANE,
) -> Road:
    """
    Create a rounded rectangle track where turn radius is automatically
    set to half the shorter side (creating semicircular ends if square,
    or the tightest possible turns otherwise).

    Args:
        center: Center of the rectangle (x, y) in meters
        length: Total length in meters
        width: Total width in meters
        lane_config: Lane configuration for the road

    Returns:
        A Road forming a rounded rectangular loop
    """
    # Use the maximum possible turn radius
    turn_radius = min(length, width) / 2 - 0.1  # Small margin for safety
    return create_rectangular_track(
        center=center,
        length=length,
        width=width,
        turn_radius=turn_radius,
        lane_config=lane_config,
    )
