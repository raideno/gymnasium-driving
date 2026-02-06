import typing
import gymnasium

import numpy as np

class WithDynamicsInfo(gymnasium.ObservationWrapper):
    """
    Adds vehicle dynamics information to observations.
    
    Provides:
    - Linear acceleration (longitudinal and lateral)
    - Angular velocity (yaw rate)
    - Slip angle
    - Jerk (rate of change of acceleration)
    - Steering rate
    
    Args:
        env: The environment to wrap
        include_history: Number of past states to track for derivatives
    """
    
    def __init__(
        self,
        env: gymnasium.Env,
        include_history: int = 3,
    ):
        super().__init__(env)
        
        self.include_history = include_history
        
        new_spaces = dict(self.observation_space.spaces)
        # dynamics: [accel_long, accel_lat, yaw_rate, slip_angle, jerk, steering_rate]
        new_spaces["vehicle/dynamics"] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )
        # velocity components [vx, vy] in world frame
        new_spaces["vehicle/velocity_components"] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = gymnasium.spaces.Dict(new_spaces)
        
        self._velocity_history: typing.List[float] = []
        self._heading_history: typing.List[float] = []
        self._steering_history: typing.List[float] = []
        self._accel_history: typing.List[float] = []
    
    def observation(self, observation: dict) -> dict:
        heading = self.env.unwrapped.state["yaw"]
        velocity = self.env.unwrapped.state["velocity"]
        
        vx = velocity * np.cos(heading)
        vy = velocity * np.sin(heading)
        
        self._velocity_history.append(velocity)
        self._heading_history.append(heading)
        
        if len(self._velocity_history) > self.include_history:
            self._velocity_history.pop(0)
        if len(self._heading_history) > self.include_history:
            self._heading_history.pop(0)
        
        # Compute derivatives
        dt = self.env.unwrapped.DELTA_TIME
        
        # longitudinal acceleration
        if len(self._velocity_history) >= 2:
            accel_long = (self._velocity_history[-1] - self._velocity_history[-2]) / dt
        else:
            accel_long = 0.0
        
        # yaw rate
        if len(self._heading_history) >= 2:
            heading_diff = self._heading_history[-1] - self._heading_history[-2]
            # Normalize heading difference
            heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
            yaw_rate = heading_diff / dt
        else:
            yaw_rate = 0.0
        
        # lateral acceleration (centripetal)
        accel_lat = velocity * yaw_rate
        
        # slip angle (simplified - assuming bicycle model)
        wheelbase = getattr(self.env, 'WHEELBASE', 2.5)
        if abs(velocity) > 0.1 and abs(yaw_rate) > 0.001:
            # Approximate slip angle from yaw rate and velocity
            turn_radius = velocity / yaw_rate if abs(yaw_rate) > 0.001 else float('inf')
            slip_angle = np.arctan(wheelbase / (2 * abs(turn_radius))) if abs(turn_radius) > 0.1 else 0.0
            slip_angle *= np.sign(yaw_rate)
        else:
            slip_angle = 0.0
        
        # jerk (rate of change of acceleration)
        self._accel_history.append(accel_long)
        if len(self._accel_history) > self.include_history:
            self._accel_history.pop(0)
        
        if len(self._accel_history) >= 2:
            jerk = (self._accel_history[-1] - self._accel_history[-2]) / dt
        else:
            jerk = 0.0
        
        # steering rate
        steering_rate = 0.0
        if hasattr(self.env, '_episode_data'):
            steering_angles = self.env.unwrapped._episode_data.get('steering_angles', [])
            if len(steering_angles) >= 2:
                steering_rate = (steering_angles[-1] - steering_angles[-2]) / dt
        
        dynamics = np.array([
            accel_long,
            accel_lat,
            yaw_rate,
            slip_angle,
            jerk,
            steering_rate,
        ], dtype=np.float32)
        
        observation["vehicle/dynamics"] = dynamics
        observation["vehicle/velocity_components"] = np.array([vx, vy], dtype=np.float32)
        
        return observation
    
    def reset(self, **kwargs):
        self._velocity_history.clear()
        self._heading_history.clear()
        self._steering_history.clear()
        self._accel_history.clear()
        return super().reset(**kwargs)
