import typing
import pygame

import numpy as np

import environments.bicycle.components.constants as constants

class Renderer:
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
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(self.screen_size)
                pygame.display.set_caption("Bicycle Car Environment")
            else:
                self.screen = pygame.Surface(self.screen_size)
            self.clock = pygame.time.Clock()
    
    def render_frame(self, env) -> np.ndarray | None:
        """
        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        self.initialize()
        
        self.screen.fill(constants.light_gray)
        
        self._draw_grid(env)
        
        if env.road_network is not None:
            self._draw_roads(env)
        
        if env.global_path is not None and len(env.global_path) > 1:
            self._draw_global_path(env)
        
        self._draw_obstacles(env)
        
        self._draw_goal(env)
        
        if env.state is not None:
            self._draw_car(env)
        
        self._draw_spawn(env)
        
        self._draw_scale_indicator(env)
        
        self._draw_info_text(env)
        
        env.overlay_manager.render(self.screen, env._world_to_screen, env._meters_to_pixels)

        env.performance_tracker.render(self.screen, self.screen_size, env.simulation_time)
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.render_fps)
            return None
        
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )
    
    def _draw_grid(self, env) -> None:
        grid_spacing = 10.0  # meters
        grid_color = (200, 200, 200)
        
        min_x, min_y = env.world_origin
        max_x = min_x + env.world_size[0]
        max_y = min_y + env.world_size[1]
        
        # Vertical grid lines
        x = np.ceil(min_x / grid_spacing) * grid_spacing
        while x <= max_x:
            start = env._world_to_screen((x, min_y))
            end = env._world_to_screen((x, max_y))
            pygame.draw.line(self.screen, grid_color, start, end, 1)
            x += grid_spacing
        
        # Horizontal grid lines
        y = np.ceil(min_y / grid_spacing) * grid_spacing
        while y <= max_y:
            start = env._world_to_screen((min_x, y))
            end = env._world_to_screen((max_x, y))
            pygame.draw.line(self.screen, grid_color, start, end, 1)
            y += grid_spacing
    
    def _draw_roads(self, env) -> None:
        """Draw road network."""
        # First pass: draw all road surfaces (polygons)
        for road in env.road_network.roads:
            half_width = road.lane_config.half_width
            
            for segment in road.segments:
                num_points = max(20, int(segment.get_length() * 2))
                left, right = segment.get_boundary_points(half_width, num_points)
                
                left_screen = [env._world_to_screen(p) for p in left]
                right_screen = [env._world_to_screen(p) for p in right]
                
                polygon_points = left_screen + right_screen[::-1]
                if len(polygon_points) >= 3:
                    pygame.draw.polygon(self.screen, constants.gray, polygon_points)
        
        # Second pass: draw boundary lines and center lines
        for road in env.road_network.roads:
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
                
                left_screen = [env._world_to_screen(p) for p in left_trimmed]
                right_screen = [env._world_to_screen(p) for p in right_trimmed]
                
                if len(left_screen) >= 2:
                    pygame.draw.lines(self.screen, constants.dark_gray, False, left_screen, 3)
                    pygame.draw.lines(self.screen, constants.dark_gray, False, right_screen, 3)
                
                centerline = segment.get_centerline_points(num_points)
                if num_lanes > 1:
                    center_screen = [env._world_to_screen(p) for p in centerline]
                    for i in range(0, len(center_screen) - 1, 4):
                        end_idx = min(i + 2, len(center_screen) - 1)
                        pygame.draw.line(
                            self.screen,
                            constants.yellow,
                            center_screen[i],
                            center_screen[end_idx],
                            2,
                        )
    
    def _draw_global_path(self, env) -> None:
        """Draw global path."""
        global_path = env.global_path
        path_screen = [env._world_to_screen(p) for p in global_path]
        # Draw path line (closed loop if it's a circular path)
        is_loop = True  # Assume loop by default
        pygame.draw.lines(self.screen, (150, 100, 200), is_loop, path_screen, 2)
        # Draw waypoints as small circles (only every few points for cleaner look)
        step = max(1, len(path_screen) // 128)  # Show ~30 waypoint markers max
        for i in range(0, len(path_screen), step):
            point = path_screen[i]
            # Color start point constants.green, others purple
            if i == 0:
                color = (100, 200, 100)
                radius = 6
            else:
                color = (150, 100, 200)
                radius = 3
            pygame.draw.circle(self.screen, color, point, radius)
    
    def _draw_obstacles(self, env) -> None:
        for obs in env.obstacles:
            center_screen = env._world_to_screen(obs.center)
            # Import here to avoid circular dependency
            from ..components.obstacles import Circle, Rectangle
            if isinstance(obs, Circle):
                radius_px = env._meters_to_pixels(obs.radius)
                pygame.draw.circle(self.screen, constants.red, center_screen, radius_px)
                pygame.draw.circle(self.screen, constants.dark_gray, center_screen, radius_px, 2)
            elif isinstance(obs, Rectangle):
                width_px = env._meters_to_pixels(obs.width)
                height_px = env._meters_to_pixels(obs.height)
                rect = pygame.Rect(
                    center_screen[0] - width_px // 2,
                    center_screen[1] - height_px // 2,
                    width_px,
                    height_px,
                )
                pygame.draw.rect(self.screen, constants.red, rect)
                pygame.draw.rect(self.screen, constants.dark_gray, rect, 2)
    
    def _draw_goal(self, env) -> None:
        goal_screen = env._world_to_screen(env.goal_pos)
        goal_radius_px = env._meters_to_pixels(env.goal_radius)
        pygame.draw.circle(self.screen, constants.green, goal_screen, goal_radius_px)
        pygame.draw.circle(self.screen, constants.dark_gray, goal_screen, goal_radius_px, 2)
    
    def _draw_car(self, env) -> None:
        """Draw car as a proper rectangle with heading."""
        state = env.state
        x, y, theta, v = state
        
        # Use instance car dimensions
        half_length = env.CAR_LENGTH / 2
        half_width = env.CAR_WIDTH / 2
        
        # Get car corners using the helper method
        corners_world = env._get_car_corners()
        
        # Convert to screen coordinates
        corners_screen = [env._world_to_screen(corner) for corner in corners_world]
        
        # Draw car body
        pygame.draw.polygon(self.screen, constants.blue, corners_screen)
        pygame.draw.polygon(self.screen, constants.dark_gray, corners_screen, 2)
        
        # Draw front indicator (small triangle at front)
        pos = np.array([x, y])
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        front_center = pos + rot @ np.array([half_length, 0])
        front_left = pos + rot @ np.array([half_length - 0.5, half_width * 0.5])
        front_right = pos + rot @ np.array([half_length - 0.5, -half_width * 0.5])
        front_points = [
            env._world_to_screen(front_center),
            env._world_to_screen(front_left),
            env._world_to_screen(front_right),
        ]
        pygame.draw.polygon(self.screen, (100, 100, 220), front_points)
    
    def _draw_spawn(self, env) -> None:
        """Draw spawn marker."""
        spawn_screen = env._world_to_screen(env.spawn_pos)
        pygame.draw.circle(self.screen, constants.black, spawn_screen, 5)
        pygame.draw.circle(self.screen, constants.white, spawn_screen, 3)
    
    def _draw_scale_indicator(self, env) -> None:
        """Draw a scale bar showing real-world distance."""
        # Draw a 10-meter scale bar in the bottom-right corner
        scale_length_m = 10.0
        scale_length_px = env._meters_to_pixels(scale_length_m)
        
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
    
    def _draw_info_text(self, env) -> None:
        """Draw current state information."""
        state = env.state
        if state is None:
            return
        
        font = pygame.font.Font(None, 24)
        x, y, theta, v = state
        
        lines = [
            f"Position: ({x:.1f}m, {y:.1f}m)",
            f"Heading: {np.degrees(theta):.1f} deg",
            f"Velocity: {v:.1f} m/s ({v * 3.6:.1f} km/h)",
        ]
        
        if env.road_network is not None:
            on_road = not env.road_network.is_off_road(state[:2])
            lines.append(f"On road: {'Yes' if on_road else 'No'}")
        
        for i, line in enumerate(lines):
            text = font.render(line, True, (0, 0, 0))
            self.screen.blit(text, (10, 10 + i * 22))
    
    def close(self) -> None:
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
