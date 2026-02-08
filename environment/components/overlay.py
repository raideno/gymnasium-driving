import typing
import pygame

import numpy as np

class OverlayManager:
    def __init__(self):
        self._overlays: typing.List[typing.Dict[str, typing.Any]] = []
    
    def clear(self) -> None:
        self._overlays.clear()
    
    def add_circle(
        self,
        center: typing.Tuple[float, float],
        radius: float,
        color: typing.Tuple[int, int, int] = (255, 0, 0),
        width: int = 0,
    ) -> None:
        self._overlays.append({
            "type": "circle",
            "center": center,
            "radius": radius,
            "color": color,
            "width": width,
        })
    
    def add_point(
        self,
        position: typing.Tuple[float, float],
        color: typing.Tuple[int, int, int] = (255, 0, 0),
        size: int = 5,
    ) -> None:
        self._overlays.append({
            "type": "point",
            "position": position,
            "color": color,
            "size": size,
        })
    
    def add_line(
        self,
        start: typing.Tuple[float, float],
        end: typing.Tuple[float, float],
        color: typing.Tuple[int, int, int] = (255, 0, 0),
        width: int = 2,
    ) -> None:
        self._overlays.append({
            "type": "line",
            "start": start,
            "end": end,
            "color": color,
            "width": width,
        })
    
    def add_path(
        self,
        points: np.ndarray,
        color: typing.Tuple[int, int, int] = (255, 0, 0),
        width: int = 2,
        closed: bool = False,
    ) -> None:
        self._overlays.append({
            "type": "path",
            "points": points,
            "color": color,
            "width": width,
            "closed": closed,
        })
    
    def add_arrow(
        self,
        start: typing.Tuple[float, float],
        end: typing.Tuple[float, float],
        color: typing.Tuple[int, int, int] = (255, 0, 0),
        width: int = 2,
        head_size: float = 0.5,
    ) -> None:
        self._overlays.append({
            "type": "arrow",
            "start": start,
            "end": end,
            "color": color,
            "width": width,
            "head_size": head_size,
        })
    
    def add_text(
        self,
        position: typing.Tuple[float, float],
        text: str,
        color: typing.Tuple[int, int, int] = (0, 0, 0),
        font_size: int = 20,
    ) -> None:
        self._overlays.append({
            "type": "text",
            "position": position,
            "text": text,
            "color": color,
            "font_size": font_size,
        })
    
    def render(
        self,
        screen: pygame.Surface,
        world_to_screen: typing.Callable,
        meters_to_pixels: typing.Callable,
    ) -> None:
        """
        Render all overlays on the screen.
        
        Args:
            screen: Pygame surface to draw on
            world_to_screen: Function to convert world coords to screen coords
            meters_to_pixels: Function to convert meters to pixels
        """
        for overlay in self._overlays:
            overlay_type = overlay["type"]
            
            if overlay_type == "circle":
                center_screen = world_to_screen(overlay["center"])
                radius_px = meters_to_pixels(overlay["radius"])
                pygame.draw.circle(
                    screen,
                    overlay["color"],
                    center_screen,
                    max(1, radius_px),
                    overlay["width"],
                )
            
            elif overlay_type == "point":
                pos_screen = world_to_screen(overlay["position"])
                pygame.draw.circle(
                    screen,
                    overlay["color"],
                    pos_screen,
                    overlay["size"],
                )
            
            elif overlay_type == "line":
                start_screen = world_to_screen(overlay["start"])
                end_screen = world_to_screen(overlay["end"])
                pygame.draw.line(
                    screen,
                    overlay["color"],
                    start_screen,
                    end_screen,
                    overlay["width"],
                )
            
            elif overlay_type == "path":
                points = overlay["points"]
                if len(points) >= 2:
                    points_screen = [world_to_screen(p) for p in points]
                    pygame.draw.lines(
                        screen,
                        overlay["color"],
                        overlay["closed"],
                        points_screen,
                        overlay["width"],
                    )
            
            elif overlay_type == "arrow":
                start = np.array(overlay["start"])
                end = np.array(overlay["end"])
                start_screen = world_to_screen(start)
                end_screen = world_to_screen(end)
                
                # Main Line
                pygame.draw.line(
                    screen,
                    overlay["color"],
                    start_screen,
                    end_screen,
                    overlay["width"],
                )
                
                # Arrow Head
                direction = end - start
                length = np.linalg.norm(direction)
                if length > 0.01:
                    direction = direction / length
                    perpendicular = np.array([-direction[1], direction[0]])
                    head_size = overlay["head_size"]
                    
                    head_left = end - direction * head_size + perpendicular * head_size * 0.5
                    head_right = end - direction * head_size - perpendicular * head_size * 0.5
                    
                    head_points = [
                        end_screen,
                        world_to_screen(head_left),
                        world_to_screen(head_right),
                    ]
                    pygame.draw.polygon(screen, overlay["color"], head_points)
            
            elif overlay_type == "text":
                pos_screen = world_to_screen(overlay["position"])
                font = pygame.font.Font(None, overlay["font_size"])
                text_surface = font.render(overlay["text"], True, overlay["color"])
                screen.blit(text_surface, pos_screen)
