import typing

from kbm import KinematicBicycleModel

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
        
        self._model = KinematicBicycleModel(
            wheelbase=wheelbase,
            max_steer=max_steer,
            delta_time=delta_time,
        )
    
    def compute_state(
        self,
        x: float,
        y: float,
        yaw: float,
        steer: float,
        velocity: float,
        acceleration: float,
    ) -> typing.Dict[str, float]:
        """
        Returns:
            Dictionary with keys: 'x', 'y', 'yaw', 'velocity'
        """
        return self._model.compute_state(
            x=x,
            y=y,
            yaw=yaw,
            steer=steer,
            velocity=velocity,
            acceleration=acceleration,
        )