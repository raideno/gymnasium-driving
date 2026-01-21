import time
import typing
import pygame

class PerformanceTracker:
    """
    Tracks and displays performance metrics like FPS, simulation time, and step count.
    """
    
    def __init__(self, show_performance: bool = True):
        self._start_time: float | None = None
        self._last_step_time: float | None = None
        self._fps: float = 0.0
        self._step_count: int = 0
        self._show_performance: bool = show_performance
    
    def reset(self) -> None:
        self._start_time = time.time()
        self._last_step_time = self._start_time
        self._fps = 0.0
        self._step_count = 0
    
    def update(self) -> None:
        current_time = time.time()
        self._step_count += 1
        
        if self._last_step_time is not None:
            frame_delta = current_time - self._last_step_time
            if frame_delta > 0:
                # NOTE: exponentially moving average for smooth FPS
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / frame_delta)
        
        self._last_step_time = current_time
    
    def render(
        self,
        screen: pygame.Surface,
        screen_size: typing.Tuple[int, int],
        sim_time: float,
    ) -> None:
        """
        Draw FPS and elapsed time in the top-right corner.
        
        Args:
            screen: Pygame surface to draw on
            screen_size: Screen dimensions (width, height)
            sim_time: Current simulation time
        """
        if not self._show_performance or screen is None:
            return
        
        font = pygame.font.Font(None, 24)
        
        lines = [
            f"FPS: {self.fps:.1f}",
            f"Sim: {sim_time:.2f}s",
            f"Real: {self.elapsed_time:.2f}s",
            f"Steps: {self._step_count}",
        ]
        
        for i, line in enumerate(lines):
            text = font.render(line, True, (0, 0, 0))
            text_rect = text.get_rect()
            text_rect.topright = (screen_size[0] - 10, 10 + i * 22)
            screen.blit(text, text_rect)
    
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def elapsed_time(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
    
    @property
    def step_count(self) -> int:
        return self._step_count
    
    def show_performance(self, show: bool = True) -> None:
        self._show_performance = show
