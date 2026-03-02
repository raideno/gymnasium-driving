import numpy as np


class AckermannModel:
    def __init__(
        self,
        wheelbase: float,
        track_width: float,
        max_steer: float,
        delta_time: float,
    ):
        """
        Args:
            wheelbase: Distance between front and rear axles (meters)
            track_width: Distance between left and right wheels (meters)
            max_steer: Maximum steering angle (radians)
            delta_time: Time step for integration (seconds)
        """
        self.wheelbase = wheelbase
        self.track_width = track_width
        self.max_steer = max_steer
        self.delta_time = delta_time

    def compute_wheel_angles(self, steer: float) -> tuple[float, float]:
        if abs(steer) < 1e-6:
            return (0.0, 0.0)

        R = self.wheelbase / np.tan(abs(steer))

        inner_angle = np.arctan(self.wheelbase / (R - self.track_width / 2))
        outer_angle = np.arctan(self.wheelbase / (R + self.track_width / 2))

        sign = np.sign(steer)

        return (sign * inner_angle, sign * outer_angle)

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

        if abs(steer) < 1e-6:
            x_next = x + velocity_avg * np.cos(yaw) * self.delta_time
            y_next = y + velocity_avg * np.sin(yaw) * self.delta_time
            yaw_next = yaw
        else:
            omega = velocity_avg * np.tan(steer) / self.wheelbase

            yaw_next = yaw + omega * self.delta_time

            R = self.wheelbase / np.tan(abs(steer))

            cx = x - R * np.sin(yaw) * np.sign(steer)
            cy = y + R * np.cos(yaw) * np.sign(steer)

            x_next = cx + R * np.sin(yaw_next) * np.sign(steer)
            y_next = cy - R * np.cos(yaw_next) * np.sign(steer)

        yaw_next = np.arctan2(np.sin(yaw_next), np.cos(yaw_next))

        return {
            "x": float(x_next),
            "y": float(y_next),
            "yaw": float(yaw_next),
            "velocity": float(velocity_next),
        }
