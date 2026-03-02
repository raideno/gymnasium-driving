import numpy as np


class BicycleModel:
    def __init__(
        self,
        wheelbase: float,
        max_steer: float,
        delta_time: float,
    ):
        """
        Args:
            wheelbase: Distance between front and rear axles (meters)
            max_steer: Maximum steering angle (radians)
            delta_time: Time step for integration (seconds)
        """
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.delta_time = delta_time

    def compute_state(
        self,
        x: float,
        y: float,
        yaw: float,
        steer: float,
        velocity: float,
        acceleration: float,
    ):
        """
        Returns:
            Dictionary with keys: 'x', 'y', 'yaw', 'velocity'
        """
        steer = np.clip(steer, -self.max_steer, self.max_steer)

        velocity_next = velocity + acceleration * self.delta_time
        velocity_avg = (velocity + velocity_next) / 2

        x_next = x + velocity_avg * np.cos(yaw) * self.delta_time
        y_next = y + velocity_avg * np.sin(yaw) * self.delta_time
        yaw_next = yaw + velocity_avg * np.tan(steer) / self.wheelbase * self.delta_time
        yaw_next = np.arctan2(np.sin(yaw_next), np.cos(yaw_next))

        return {
            "x": float(x_next),
            "y": float(y_next),
            "yaw": float(yaw_next),
            "velocity": float(velocity_next),
        }
