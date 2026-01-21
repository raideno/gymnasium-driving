import typing
import pygame

import numpy as np

class Renderer:
    """
    Handles all rendering operations for the bicycle car environment.
    """
    
    def __init__(
        self,
        screen_size: typing.Tuple[int, int],
        render_mode: str | None,
        render_fps: int = 30,
    ):
        self.screen_size = screen_size
        self.render_mode = render_mode
        self.render_fps = render_fps
        
        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
    
    def initialize(self) -> None:
        """Initialize pygame and create the screen."""
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(self.screen_size)
                pygame.display.set_caption("Bicycle Car Environment")
            else:
                self.screen = pygame.Surface(self.screen_size)
            self.clock = pygame.time.Clock()
    
    def render_frame(
        self,
        state: np.ndarray | None,
        spawn_pos: np.ndarray,
        goal_pos: np.ndarray,
        goal_radius: float,
        obstacles: list,
        road_network,
        global_path: np.ndarray | None,
        car_length: float,
        car_width: float,
        world_origin: np.ndarray,
        world_size: np.ndarray,
        pixels_per_meter: float,
        world_to_screen: typing.Callable,
        meters_to_pixels: typing.Callable,
        get_car_corners: typing.Callable,
        sim_time: float,
        overlay_manager,
        performance_tracker,
    ) -> np.ndarray | None:
        """
        Render a single frame with proper meter-based scaling.
        
        Args:
            state: Current state [x, y, theta, v]
            spawn_pos: Spawn position
            goal_pos: Goal position
            goal_radius: Goal radius
            obstacles: List of obstacles
            road_network: Road network object
            global_path: Global path waypoints
            car_length: Car length
            car_width: Car width
            world_origin: World origin coordinates
            world_size: World size
            pixels_per_meter: Scaling factor
            world_to_screen: Function to convert world to screen coords
            meters_to_pixels: Function to convert meters to pixels
            get_car_corners: Function to get car corners
            sim_time: Current simulation time
            overlay_manager: OverlayManager instance
            performance_tracker: PerformanceTracker instance
            
        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        self.initialize()
        
        # Colors
        white = (255, 255, 255)
        black = (0, 0, 0)
        red = (220, 60, 60)
        green = (60, 180, 60)
        blue = (60, 60, 220)
        gray = (140, 140, 140)
        dark_gray = (80, 80, 80)
        yellow = (255, 220, 50)
        light_gray = (220, 220, 220)
        
        # Clear screen
        self.screen.fill(light_gray)
        
        # Draw grid
        self._draw_grid(world_origin, world_size, world_to_screen)
        
        # Draw roads
        if road_network is not None:
            self._draw_roads(road_network, world_to_screen, gray, dark_gray, yellow)
        
        # Draw global path
        if global_path is not None and len(global_path) > 1:
            self._draw_global_path(global_path, world_to_screen)
        
        # Draw obstacles
        self._draw_obstacles(obstacles, world_to_screen, meters_to_pixels, red, dark_gray)
        
        # Draw goal
        self._draw_goal(goal_pos, goal_radius, world_to_screen, meters_to_pixels, green, dark_gray)
        
        # Draw car
        if state is not None:
            self._draw_car(state, car_length, car_width, world_to_screen, get_car_corners, blue, dark_gray)
        
        # Draw spawn marker
        self._draw_spawn(spawn_pos, world_to_screen, black, white)
        
        # Draw custom overlays
        overlay_manager.render(self.screen, world_to_screen, meters_to_pixels)
        
        # Draw scale indicator
        self._draw_scale_indicator(meters_to_pixels)
        
        # Draw info text
        self._draw_info_text(state, road_network)
        
        # Draw performance info
        performance_tracker.render(self.screen, self.screen_size, sim_time)
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.render_fps)
            return None
        
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )
    
    def _draw_grid(
        self,
        world_origin: np.ndarray,
        world_size: np.ndarray,
        world_to_screen: typing.Callable,
    ) -> None:
        """Draw grid for scale reference."""
        grid_spacing = 10.0  # meters
        grid_color = (200, 200, 200)
        
        min_x, min_y = world_origin
        max_x = min_x + world_size[0]
        max_y = min_y + world_size[1]
        
        # Vertical grid lines
        x = np.ceil(min_x / grid_spacing) * grid_spacing
        while x <= max_x:
            start = world_to_screen((x, min_y))
            end = world_to_screen((x, max_y))
            pygame.draw.line(self.screen, grid_color, start, end, 1)
            x += grid_spacing
        
        # Horizontal grid lines
        y = np.ceil(min_y / grid_spacing) * grid_spacing
        while y <= max_y:
            start = world_to_screen((min_x, y))
            end = world_to_screen((max_x, y))
            pygame.draw.line(self.screen, grid_color, start, end, 1)
            y += grid_spacing
    
    def _draw_roads(
        self,
        road_network,
        world_to_screen: typing.Callable,
        gray: tuple,
        dark_gray: tuple,
        yellow: tuple,
    ) -> None:
        """Draw road network."""
        # First pass: draw all road surfaces (polygons)
        for road in road_network.roads:
            half_width = road.lane_config.half_width
            
            for segment in road.segments:
                num_points = max(20, int(segment.get_length() * 2))
                left, right = segment.get_boundary_points(half_width, num_points)
                
                left_screen = [world_to_screen(p) for p in left]
                right_screen = [world_to_screen(p) for p in right]
                
                polygon_points = left_screen + right_screen[::-1]
                if len(polygon_points) >= 3:
                    pygame.draw.polygon(self.screen, gray, polygon_points)
        
        # Second pass: draw boundary lines and center lines
        for road in road_network.roads:
            half_width = road.lane_config.half_width
            num_lanes = road.lane_config.num_lanes
            total_length = road.get_total_length()
            
            # Calculate trim amounts in terms of fraction of total length
            start_trim_frac = road.start_trim / total_length if total_length > 0 else 0
            end_trim_frac = road.end_trim / total_length if total_length > 0 else 0
            
            for seg_idx, segment in enumerate(road.segments):
                seg_length = segment.get_length()
                num_points = max(20, int(seg_length * 2))
                left, right = segment.get_boundary_points(half_width, num_points)
                
                # Calculate which points to trim for this segment
                is_first_segment = seg_idx == 0
                is_last_segment = seg_idx == len(road.segments) - 1
                
                # Trim start of first segment
                start_idx = 0
                if is_first_segment and road.start_trim > 0:
                    trim_points = int(road.start_trim / seg_length * num_points)
                    start_idx = min(trim_points, num_points - 2)
                
                # Trim end of last segment
                end_idx = num_points
                if is_last_segment and road.end_trim > 0:
                    trim_points = int(road.end_trim / seg_length * num_points)
                    end_idx = max(num_points - trim_points, start_idx + 2)
                
                # Get trimmed boundary points
                left_trimmed = left[start_idx:end_idx]
                right_trimmed = right[start_idx:end_idx]
                
                left_screen = [world_to_screen(p) for p in left_trimmed]
                right_screen = [world_to_screen(p) for p in right_trimmed]
                
                if len(left_screen) >= 2:
                    pygame.draw.lines(self.screen, dark_gray, False, left_screen, 3)
                    pygame.draw.lines(self.screen, dark_gray, False, right_screen, 3)
                
                centerline = segment.get_centerline_points(num_points)
                if num_lanes > 1:
                    center_screen = [world_to_screen(p) for p in centerline]
                    for i in range(0, len(center_screen) - 1, 4):
                        end_idx = min(i + 2, len(center_screen) - 1)
                        pygame.draw.line(
                            self.screen,
                            yellow,
                            center_screen[i],
                            center_screen[end_idx],
                            2,
                        )
    
    def _draw_global_path(
        self,
        global_path: np.ndarray,
        world_to_screen: typing.Callable,
    ) -> None:
        """Draw global path."""
        path_screen = [world_to_screen(p) for p in global_path]
        # Draw path line (closed loop if it's a circular path)
        is_loop = True  # Assume loop by default
        pygame.draw.lines(self.screen, (150, 100, 200), is_loop, path_screen, 2)
        # Draw waypoints as small circles (only every few points for cleaner look)
        step = max(1, len(path_screen) // 30)  # Show ~30 waypoint markers max
        for i in range(0, len(path_screen), step):
            point = path_screen[i]
            # Color start point green, others purple
            if i == 0:
                color = (100, 200, 100)
                radius = 6
            else:
                color = (150, 100, 200)
                radius = 3
            pygame.draw.circle(self.screen, color, point, radius)
    
    def _draw_obstacles(
        self,
        obstacles: list,
        world_to_screen: typing.Callable,
        meters_to_pixels: typing.Callable,
        red: tuple,
        dark_gray: tuple,
    ) -> None:
        """Draw obstacles."""
        for obs in obstacles:
            center_screen = world_to_screen(obs.center)
            # Import here to avoid circular dependency
            from ..obstacles import Circle, Rectangle
            if isinstance(obs, Circle):
                radius_px = meters_to_pixels(obs.radius)
                pygame.draw.circle(self.screen, red, center_screen, radius_px)
                pygame.draw.circle(self.screen, dark_gray, center_screen, radius_px, 2)
            elif isinstance(obs, Rectangle):
                width_px = meters_to_pixels(obs.width)
                height_px = meters_to_pixels(obs.height)
                rect = pygame.Rect(
                    center_screen[0] - width_px // 2,
                    center_screen[1] - height_px // 2,
                    width_px,
                    height_px,
                )
                pygame.draw.rect(self.screen, red, rect)
                pygame.draw.rect(self.screen, dark_gray, rect, 2)
    
    def _draw_goal(
        self,
        goal_pos: np.ndarray,
        goal_radius: float,
        world_to_screen: typing.Callable,
        meters_to_pixels: typing.Callable,
        green: tuple,
        dark_gray: tuple,
    ) -> None:
        """Draw goal."""
        goal_screen = world_to_screen(goal_pos)
        goal_radius_px = meters_to_pixels(goal_radius)
        pygame.draw.circle(self.screen, green, goal_screen, goal_radius_px)
        pygame.draw.circle(self.screen, dark_gray, goal_screen, goal_radius_px, 2)
    
    def _draw_car(
        self,
        state: np.ndarray,
        car_length: float,
        car_width: float,
        world_to_screen: typing.Callable,
        get_car_corners: typing.Callable,
        blue: tuple,
        dark_gray: tuple,
    ) -> None:
        """Draw car as a proper rectangle with heading."""
        x, y, theta, v = state
        
        # Use instance car dimensions
        half_length = car_length / 2
        half_width = car_width / 2
        
        # Get car corners using the helper method
        corners_world = get_car_corners()
        
        # Convert to screen coordinates
        corners_screen = [world_to_screen(corner) for corner in corners_world]
        
        # Draw car body
        pygame.draw.polygon(self.screen, blue, corners_screen)
        pygame.draw.polygon(self.screen, dark_gray, corners_screen, 2)
        
        # Draw front indicator (small triangle at front)
        pos = np.array([x, y])
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        front_center = pos + rot @ np.array([half_length, 0])
        front_left = pos + rot @ np.array([half_length - 0.5, half_width * 0.5])
        front_right = pos + rot @ np.array([half_length - 0.5, -half_width * 0.5])
        front_points = [
            world_to_screen(front_center),
            world_to_screen(front_left),
            world_to_screen(front_right),
        ]
        pygame.draw.polygon(self.screen, (100, 100, 220), front_points)
    
    def _draw_spawn(
        self,
        spawn_pos: np.ndarray,
        world_to_screen: typing.Callable,
        black: tuple,
        white: tuple,
    ) -> None:
        """Draw spawn marker."""
        spawn_screen = world_to_screen(spawn_pos)
        pygame.draw.circle(self.screen, black, spawn_screen, 5)
        pygame.draw.circle(self.screen, white, spawn_screen, 3)
    
    def _draw_scale_indicator(self, meters_to_pixels: typing.Callable) -> None:
        """Draw a scale bar showing real-world distance."""
        # Draw a 10-meter scale bar in the bottom-right corner
        scale_length_m = 10.0
        scale_length_px = meters_to_pixels(scale_length_m)
        
        bar_x = self.screen_size[0] - scale_length_px - 20
        bar_y = self.screen_size[1] - 30
        
        # Draw the bar
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (bar_x, bar_y),
            (bar_x + scale_length_px, bar_y),
            3,
        )
        # End caps
        pygame.draw.line(
            self.screen, (0, 0, 0), (bar_x, bar_y - 5), (bar_x, bar_y + 5), 2
        )
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (bar_x + scale_length_px, bar_y - 5),
            (bar_x + scale_length_px, bar_y + 5),
            2,
        )
        
        # Label
        font = pygame.font.Font(None, 20)
        text = font.render(f"{scale_length_m:.0f}m", True, (0, 0, 0))
        text_rect = text.get_rect(center=(bar_x + scale_length_px // 2, bar_y - 12))
        self.screen.blit(text, text_rect)
    
    def _draw_info_text(self, state: np.ndarray | None, road_network) -> None:
        """Draw current state information."""
        if state is None:
            return
        
        font = pygame.font.Font(None, 24)
        x, y, theta, v = state
        
        lines = [
            f"Position: ({x:.1f}m, {y:.1f}m)",
            f"Heading: {np.degrees(theta):.1f} deg",
            f"Velocity: {v:.1f} m/s ({v * 3.6:.1f} km/h)",
        ]
        
        if road_network is not None:
            on_road = not road_network.is_off_road(state[:2])
            lines.append(f"On road: {'Yes' if on_road else 'No'}")
        
        for i, line in enumerate(lines):
            text = font.render(line, True, (0, 0, 0))
            self.screen.blit(text, (10, 10 + i * 22))
    
    def close(self) -> None:
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
