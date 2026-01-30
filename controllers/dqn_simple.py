"""
Simple DQN Controller for Lane Following.

A minimal, easy-to-understand implementation for learning purposes.
Start here and gradually add complexity as you learn!

Key concepts:
1. Observation Wrapper: Converts dict observations to a flat array
2. Action Wrapper: Converts discrete actions to continuous controls
3. Reward Wrapper: Defines what "good driving" means
4. DQN Controller: Uses a neural network to learn the best actions
"""

import typing
import numpy as np
import gymnasium
from pathlib import Path


# =============================================================================
# OBSERVATION WRAPPER
# =============================================================================

class SimpleObservationWrapper(gymnasium.ObservationWrapper):
    """
    Flattens the observation dict into a simple array.
    
    DQN needs a flat array, not a dictionary. We extract only the
    most important features for lane following:
    
    Output: [velocity, lateral_error, heading_error, on_road]
    """
    
    def __init__(self, env: gymnasium.Env, max_velocity: float = 15.0):
        super().__init__(env)
        self.max_velocity = max_velocity
        
        # Our observation is just 4 numbers, each in [-1, 1]
        self.observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # Store path for heading calculation
        self._path: typing.Optional[np.ndarray] = None
    
    def set_path(self, path: np.ndarray) -> None:
        """Set the reference path."""
        self._path = path
    
    def observation(self, obs: dict) -> np.ndarray:
        """Convert dict observation to flat array."""
        
        # 1. Normalize velocity to [-1, 1]
        velocity = obs["velocity"][0]
        velocity_norm = np.clip(velocity / self.max_velocity, -1, 1)
        
        # 2. Lateral error (distance from lane center, normalized)
        dist_to_center = obs["distance_to_lane_center"][0]
        lane_width = obs["lane_width"][0]
        lateral_error = np.clip(dist_to_center / (lane_width / 2 + 0.1), -1, 1)
        
        # 3. Heading error (how much we're pointing away from path)
        heading_error = self._compute_heading_error(obs)
        
        # 4. On road flag (1 = on road, 0 = off road)
        on_road = float(obs["on_road"])
        
        return np.array([
            velocity_norm,
            lateral_error,
            heading_error,
            on_road
        ], dtype=np.float32)
    
    def _compute_heading_error(self, obs: dict) -> float:
        """Compute normalized heading error relative to path."""
        if self._path is None or len(self._path) < 2:
            return 0.0
        
        position = obs["position"]
        heading = obs["heading"][0]
        
        # Find closest point on path
        distances = np.linalg.norm(self._path - position, axis=1)
        closest_idx = np.argmin(distances)
        
        # Get path direction at that point
        if closest_idx < len(self._path) - 1:
            direction = self._path[closest_idx + 1] - self._path[closest_idx]
        else:
            direction = self._path[closest_idx] - self._path[closest_idx - 1]
        
        path_heading = np.arctan2(direction[1], direction[0])
        
        # Compute error and normalize to [-1, 1]
        error = heading - path_heading
        error = np.arctan2(np.sin(error), np.cos(error))  # Wrap to [-pi, pi]
        
        return np.clip(error / np.pi, -1, 1)


# =============================================================================
# ACTION WRAPPER
# =============================================================================

class SimpleActionWrapper(gymnasium.ActionWrapper):
    """
    Converts discrete actions to continuous controls.
    
    DQN outputs a single integer (action index). We map it to
    [steering, throttle, brake, reverse] for the environment.
    
    Actions:
        0: Go straight
        1: Turn left
        2: Turn right
    """
    
    def __init__(
        self,
        env: gymnasium.Env,
        steering_angle: float = 0.3,  # radians (~17 degrees)
        throttle: float = 0.5,
    ):
        super().__init__(env)
        
        self.steering_angle = steering_angle
        self.throttle = throttle
        
        # Define our 3 simple actions
        self.actions = [
            (0.0, throttle),              # Straight
            (-steering_angle, throttle),  # Left
            (steering_angle, throttle),   # Right
        ]
        
        # DQN will output integers 0, 1, or 2
        self.action_space = gymnasium.spaces.Discrete(len(self.actions))
    
    def action(self, action: int) -> np.ndarray:
        """Convert discrete action to continuous control."""
        steering, throttle = self.actions[action]
        return np.array([steering, throttle, 0.0, 0.0], dtype=np.float32)


# =============================================================================
# REWARD WRAPPER
# =============================================================================

class SimpleRewardWrapper(gymnasium.Wrapper):
    """
    Defines the reward function for lane following.
    
    Good driving = staying centered + moving forward + staying on road
    
    Rewards:
        - Centered in lane: +1.0 (decreases as you drift)
        - Moving at target speed: +0.3
        - Going off road: -5.0
        - Collision: -10.0
    """
    
    def __init__(
        self,
        env: gymnasium.Env,
        target_velocity: float = 5.0,
    ):
        super().__init__(env)
        self.target_velocity = target_velocity
    
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # Compute our custom reward
        reward = self._compute_reward(obs, info)
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, obs: dict, info: dict) -> float:
        """Compute reward based on driving quality."""
        
        # 1. Lane centering: reward being close to center
        #    Formula: 1 / (1 + 4 * error^2) gives ~1.0 when centered
        dist_to_center = abs(obs["distance_to_lane_center"][0])
        centering_reward = 1.0 / (1.0 + 4.0 * dist_to_center ** 2)
        
        # 2. Speed reward: encourage target velocity
        velocity = obs["velocity"][0]
        speed_ratio = velocity / self.target_velocity if self.target_velocity > 0 else 0
        speed_reward = 0.3 * min(speed_ratio, 1.0)
        
        # 3. Combine base rewards
        reward = centering_reward + speed_reward
        
        # 4. Penalties for bad behavior
        if info.get("collision", False):
            reward = -10.0
        elif not obs["on_road"]:
            reward = -5.0
        
        return reward


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================

def make_simple_dqn_env(env: gymnasium.Env) -> gymnasium.Env:
    """
    Wrap an environment for simple DQN training.
    
    Order matters! We apply wrappers from inside to outside:
    1. Reward wrapper (needs dict obs)
    2. Observation wrapper (flattens to array)
    3. Action wrapper (discretizes actions)
    """
    env = SimpleRewardWrapper(env)
    env = SimpleObservationWrapper(env)
    env = SimpleActionWrapper(env)
    return env


# =============================================================================
# DQN CONTROLLER
# =============================================================================

class SimpleDQNController:
    """
    A minimal DQN controller for lane following.
    
    Uses Stable-Baselines3 for the DQN implementation.
    You can train it on an environment, then use get_action() for inference.
    
    Example:
        controller = SimpleDQNController()
        controller.train(env, total_timesteps=10000)
        controller.save("my_model")
        
        # Later...
        controller.load("my_model")
        action = controller.get_action(observation, path)
    """
    
    def __init__(
        self,
        steering_angle: float = 0.3,
        throttle: float = 0.5,
        learning_rate: float = 1e-4,
        buffer_size: int = 50000,
        batch_size: int = 64,
        gamma: float = 0.99,
    ):
        # Action mapping (same as SimpleActionWrapper)
        self.actions = [
            (0.0, throttle),              # Straight
            (-steering_angle, throttle),  # Left
            (steering_angle, throttle),   # Right
        ]
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Model will be set after training or loading
        self.model = None
        
        # For processing observations during inference
        self._path: typing.Optional[np.ndarray] = None
        self._max_velocity = 15.0
    
    def train(
        self,
        env: gymnasium.Env,
        total_timesteps: int = 10000,
        verbose: int = 1,
    ) -> "SimpleDQNController":
        """
        Train the DQN model on the environment.
        
        Args:
            env: A gymnasium environment (will be wrapped automatically)
            total_timesteps: How many steps to train for
            verbose: 0 = silent, 1 = progress bar
        
        Returns:
            self (for chaining)
        """
        try:
            from stable_baselines3 import DQN
        except ImportError:
            raise ImportError(
                "Please install stable-baselines3: pip install stable-baselines3"
            )
        
        # Wrap the environment
        wrapped_env = make_simple_dqn_env(env)
        
        # Create and train the model
        self.model = DQN(
            "MlpPolicy",
            wrapped_env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            gamma=self.gamma,
            verbose=verbose,
        )
        
        self.model.learn(total_timesteps=total_timesteps)
        
        return self
    
    def get_action(
        self,
        observation: dict,
        path: np.ndarray = None,
        **kwargs  # Accept extra args for compatibility
    ) -> np.ndarray:
        """
        Get control action from observation.
        
        Args:
            observation: Dict observation from environment
            path: Reference path (N, 2) array
        
        Returns:
            action: [steering, throttle, brake, reverse] array
        """
        if self.model is None:
            # No model trained - just go straight
            return np.array([0.0, 0.5, 0.0, 0.0], dtype=np.float32)
        
        # Store path for observation processing
        self._path = path
        
        # Convert observation to flat array (same as SimpleObservationWrapper)
        flat_obs = self._process_observation(observation)
        
        # Get action from model
        action_idx, _ = self.model.predict(flat_obs, deterministic=True)
        
        # Convert to continuous control
        steering, throttle = self.actions[int(action_idx)]
        return np.array([steering, throttle, 0.0, 0.0], dtype=np.float32)
    
    def _process_observation(self, obs: dict) -> np.ndarray:
        """Convert dict observation to flat array for the model."""
        
        # Velocity
        velocity = obs["velocity"][0]
        velocity_norm = np.clip(velocity / self._max_velocity, -1, 1)
        
        # Lateral error
        dist_to_center = obs["distance_to_lane_center"][0]
        lane_width = obs["lane_width"][0]
        lateral_error = np.clip(dist_to_center / (lane_width / 2 + 0.1), -1, 1)
        
        # Heading error
        heading_error = 0.0
        if self._path is not None and len(self._path) >= 2:
            position = obs["position"]
            heading = obs["heading"][0]
            
            distances = np.linalg.norm(self._path - position, axis=1)
            closest_idx = np.argmin(distances)
            
            if closest_idx < len(self._path) - 1:
                direction = self._path[closest_idx + 1] - self._path[closest_idx]
            else:
                direction = self._path[closest_idx] - self._path[closest_idx - 1]
            
            path_heading = np.arctan2(direction[1], direction[0])
            error = heading - path_heading
            error = np.arctan2(np.sin(error), np.cos(error))
            heading_error = np.clip(error / np.pi, -1, 1)
        
        # On road
        on_road = float(obs["on_road"])
        
        return np.array([velocity_norm, lateral_error, heading_error, on_road], 
                       dtype=np.float32)
    
    def save(self, path: typing.Union[str, Path]) -> None:
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("No model to save. Train first!")
        self.model.save(str(path))
    
    def load(self, path: typing.Union[str, Path]) -> "SimpleDQNController":
        """Load a trained model from a file."""
        try:
            from stable_baselines3 import DQN
        except ImportError:
            raise ImportError(
                "Please install stable-baselines3: pip install stable-baselines3"
            )
        
        self.model = DQN.load(str(path))
        return self
