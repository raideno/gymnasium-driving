# roads/predefined.py

import typing

import numpy as np

from .index import *

def create_straight_road(
    start: typing.Tuple[float, float],
    heading: float,
    length: float,
    lane_config: LaneConfig = DOUBLE_LANE,
) -> Road:
    """Create a simple straight road."""
    return RoadBuilder(start, heading, lane_config).straight(length).build()


def create_l_turn(
    start: typing.Tuple[float, float],
    heading: float,
    straight_length: float,
    turn_radius: float,
    turn_direction: str = "left",
    lane_config: LaneConfig = DOUBLE_LANE,
) -> Road:
    """Create an L-shaped road with a 90-degree turn."""
    builder = RoadBuilder(start, heading, lane_config).straight(straight_length)
    if turn_direction == "left":
        builder.turn_left(turn_radius, 90)
    else:
        builder.turn_right(turn_radius, 90)
    return builder.straight(straight_length).build()


def create_s_curve(
    start: typing.Tuple[float, float],
    heading: float,
    turn_radius: float,
    turn_angle: float = 45.0,
    straight_between: float = 10.0,
    lane_config: LaneConfig = DOUBLE_LANE,
) -> Road:
    """Create an S-curve road."""
    return (
        RoadBuilder(start, heading, lane_config)
        .turn_left(turn_radius, turn_angle)
        .straight(straight_between)
        .turn_right(turn_radius, turn_angle)
        .build()
    )


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


# =============================================================================
# Rectangular Track
# =============================================================================

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
    
def create_rectangular_track_with_entrance(
    center: typing.Tuple[float, float],
    length: float,
    width: float,
    turn_radius: float,
    entrance_side: str = "bottom",
    entrance_position: float = 0.5,
    entrance_length: float = 20.0,
    lane_config: LaneConfig = DOUBLE_LANE,
) -> RoadNetwork:
    """
    Create a rectangular track with an entrance road on one side.

    The entrance creates a T-intersection where a straight road
    connects to the track.

    Args:
        center: Center of the rectangle (x, y) in meters
        length: Length of the long sides (horizontal) in meters
        width: Length of the short sides (vertical) in meters
        turn_radius: Radius of the corner turns in meters
        entrance_side: Which side to place the entrance
            ("bottom", "top", "left", "right")
        entrance_position: Position along the side (0.0 to 1.0)
            0.0 = start of side, 0.5 = middle, 1.0 = end of side
        entrance_length: Length of the entrance road in meters
        lane_config: Lane configuration for all roads

    Returns:
        A RoadNetwork containing the track and entrance road
    """
    # Validate turn radius
    straight_long = length - 2 * turn_radius
    straight_short = width - 2 * turn_radius

    if straight_long <= 0 or straight_short <= 0:
        raise ValueError(
            f"Turn radius {turn_radius}m is too large for rectangle "
            f"{length}m x {width}m. Max turn radius: "
            f"{min(length, width) / 2}m"
        )

    # Calculate rectangle corners (inner edges of straights)
    half_length = length / 2
    half_width = width / 2
    cx, cy = center

    # Create the main rectangular track
    track = create_rectangular_track(
        center=center,
        length=length,
        width=width,
        turn_radius=turn_radius,
        lane_config=lane_config,
    )

    # Calculate entrance road position and heading based on side
    # The entrance connects from outside the track going inward
    if entrance_side == "bottom":
        # Bottom side: y = cy - half_width, x varies
        side_start_x = cx - half_length + turn_radius
        side_end_x = cx + half_length - turn_radius
        side_length = side_end_x - side_start_x

        entrance_x = side_start_x + entrance_position * side_length
        entrance_y = cy - half_width

        # Entrance starts outside and goes up (into the track)
        road_start = (entrance_x, entrance_y - entrance_length)
        road_heading = np.pi / 2  # pointing up

    elif entrance_side == "top":
        # Top side: y = cy + half_width, x varies
        side_start_x = cx + half_length - turn_radius
        side_end_x = cx - half_length + turn_radius
        side_length = abs(side_end_x - side_start_x)

        entrance_x = side_start_x - entrance_position * side_length
        entrance_y = cy + half_width

        # Entrance starts outside and goes down (into the track)
        road_start = (entrance_x, entrance_y + entrance_length)
        road_heading = -np.pi / 2  # pointing down

    elif entrance_side == "left":
        # Left side: x = cx - half_length, y varies
        side_start_y = cy + half_width - turn_radius
        side_end_y = cy - half_width + turn_radius
        side_length = abs(side_end_y - side_start_y)

        entrance_x = cx - half_length
        entrance_y = side_start_y - entrance_position * side_length

        # Entrance starts outside and goes right (into the track)
        road_start = (entrance_x - entrance_length, entrance_y)
        road_heading = 0.0  # pointing right

    elif entrance_side == "right":
        # Right side: x = cx + half_length, y varies
        side_start_y = cy - half_width + turn_radius
        side_end_y = cy + half_width - turn_radius
        side_length = abs(side_end_y - side_start_y)

        entrance_x = cx + half_length
        entrance_y = side_start_y + entrance_position * side_length

        # Entrance starts outside and goes left (into the track)
        road_start = (entrance_x + entrance_length, entrance_y)
        road_heading = np.pi  # pointing left

    else:
        raise ValueError(
            f"Unknown entrance_side: {entrance_side}. "
            f"Use 'bottom', 'top', 'left', or 'right'."
        )

    # Create the entrance road with end_trim to prevent boundary overlap at junction
    # The trim amount should be at least the half-width of the road to fully clear the junction
    entrance_road = (
        RoadBuilder(road_start, road_heading, lane_config)
        .straight(entrance_length)
        .build()
    )
    # Trim the end of the entrance road where it meets the track
    entrance_road.end_trim = lane_config.half_width

    # Create the network with both roads
    network = RoadNetwork()
    network.add_road(track)
    network.add_road(entrance_road)

    return network


def create_rectangular_track_with_pit_lane(
    center: typing.Tuple[float, float],
    length: float,
    width: float,
    turn_radius: float,
    pit_side: str = "bottom",
    pit_offset: float = 10.0,
    pit_length_ratio: float = 0.6,
    lane_config: LaneConfig = DOUBLE_LANE,
) -> RoadNetwork:
    """
    Create a rectangular track with a parallel pit lane.

    The pit lane runs parallel to one side of the track with
    entry and exit ramps.

    Args:
        center: Center of the rectangle (x, y) in meters
        length: Length of the long sides in meters
        width: Length of the short sides in meters
        turn_radius: Radius of the corner turns in meters
        pit_side: Which side to place the pit lane ("bottom", "top")
        pit_offset: Distance from track edge to pit lane center in meters
        pit_length_ratio: Pit lane length as ratio of side length (0.0-1.0)
        lane_config: Lane configuration for all roads

    Returns:
        A RoadNetwork containing the track and pit lane
    """
    half_length = length / 2
    half_width = width / 2
    cx, cy = center

    straight_long = length - 2 * turn_radius

    # Create main track
    track = create_rectangular_track(
        center=center,
        length=length,
        width=width,
        turn_radius=turn_radius,
        lane_config=lane_config,
    )

    # Calculate pit lane geometry
    pit_straight_length = straight_long * pit_length_ratio
    ramp_length = (straight_long - pit_straight_length) / 2
    ramp_angle = 15.0  # degrees

    if pit_side == "bottom":
        track_y = cy - half_width
        pit_y = track_y - pit_offset

        # Pit entry point (right side of track, going left)
        entry_x = cx + half_length - turn_radius - ramp_length * 0.2
        entry_start = (entry_x, track_y)

        # Build pit lane: ramp down -> straight -> ramp up
        pit_road = (
            RoadBuilder(entry_start, np.pi, lane_config)
            .turn_right(pit_offset / np.sin(np.radians(ramp_angle)), ramp_angle)
            .straight(pit_straight_length)
            .turn_left(pit_offset / np.sin(np.radians(ramp_angle)), ramp_angle)
            .build()
        )
        # Trim both ends where pit lane connects to track
        pit_road.start_trim = lane_config.half_width
        pit_road.end_trim = lane_config.half_width

    elif pit_side == "top":
        track_y = cy + half_width
        pit_y = track_y + pit_offset

        # Pit entry point (left side of track, going right)
        entry_x = cx - half_length + turn_radius + ramp_length * 0.2
        entry_start = (entry_x, track_y)

        pit_road = (
            RoadBuilder(entry_start, 0.0, lane_config)
            .turn_right(pit_offset / np.sin(np.radians(ramp_angle)), ramp_angle)
            .straight(pit_straight_length)
            .turn_left(pit_offset / np.sin(np.radians(ramp_angle)), ramp_angle)
            .build()
        )
        # Trim both ends where pit lane connects to track
        pit_road.start_trim = lane_config.half_width
        pit_road.end_trim = lane_config.half_width

    else:
        raise ValueError(f"pit_side must be 'bottom' or 'top', got: {pit_side}")

    network = RoadNetwork()
    network.add_road(track)
    network.add_road(pit_road)

    return network


def create_rectangular_track_with_multiple_entrances(
    center: typing.Tuple[float, float],
    length: float,
    width: float,
    turn_radius: float,
    entrances: typing.List[typing.Dict[str, typing.Any]],
    lane_config: LaneConfig = DOUBLE_LANE,
) -> RoadNetwork:
    """
    Create a rectangular track with multiple entrance roads.

    Args:
        center: Center of the rectangle (x, y) in meters
        length: Length of the long sides in meters
        width: Length of the short sides in meters
        turn_radius: Radius of the corner turns in meters
        entrances: List of entrance configurations, each a dict with:
            - "side": "bottom", "top", "left", or "right"
            - "position": 0.0 to 1.0 (position along side)
            - "length": length of entrance road in meters
        lane_config: Lane configuration for all roads

    Returns:
        A RoadNetwork containing the track and all entrance roads

    Example:
        entrances = [
            {"side": "bottom", "position": 0.3, "length": 25.0},
            {"side": "right", "position": 0.5, "length": 15.0},
        ]
    """
    half_length = length / 2
    half_width = width / 2
    cx, cy = center

    straight_long = length - 2 * turn_radius
    straight_short = width - 2 * turn_radius

    if straight_long <= 0 or straight_short <= 0:
        raise ValueError(
            f"Turn radius {turn_radius}m is too large for rectangle "
            f"{length}m x {width}m."
        )

    # Create main track
    track = create_rectangular_track(
        center=center,
        length=length,
        width=width,
        turn_radius=turn_radius,
        lane_config=lane_config,
    )

    network = RoadNetwork()
    network.add_road(track)

    # Add each entrance
    for entrance_config in entrances:
        side = entrance_config.get("side", "bottom")
        position = entrance_config.get("position", 0.5)
        ent_length = entrance_config.get("length", 20.0)

        position = max(0.05, min(0.95, position))

        if side == "bottom":
            side_start_x = cx - half_length + turn_radius
            entrance_x = side_start_x + position * straight_long
            entrance_y = cy - half_width
            road_start = (entrance_x, entrance_y - ent_length)
            road_heading = np.pi / 2

        elif side == "top":
            side_start_x = cx + half_length - turn_radius
            entrance_x = side_start_x - position * straight_long
            entrance_y = cy + half_width
            road_start = (entrance_x, entrance_y + ent_length)
            road_heading = -np.pi / 2

        elif side == "left":
            side_start_y = cy + half_width - turn_radius
            entrance_x = cx - half_length
            entrance_y = side_start_y - position * straight_short
            road_start = (entrance_x - ent_length, entrance_y)
            road_heading = 0.0

        elif side == "right":
            side_start_y = cy - half_width + turn_radius
            entrance_x = cx + half_length
            entrance_y = side_start_y + position * straight_short
            road_start = (entrance_x + ent_length, entrance_y)
            road_heading = np.pi

        else:
            raise ValueError(f"Unknown side: {side}")

        entrance_road = (
            RoadBuilder(road_start, road_heading, lane_config)
            .straight(ent_length)
            .build()
        )
        # Trim the end of the entrance road where it meets the track
        entrance_road.end_trim = lane_config.half_width
        network.add_road(entrance_road)

    return network