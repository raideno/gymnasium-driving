import numpy as np

def compute_steering_smoothness(
    steering_angles: list[float],
    dt: float = 0.1,
) -> dict[str, float]:
    """
    Compute steering smoothness metrics based on steering angle derivatives.
    
    Steering smoothness is assessed using the third derivative of the steering
    angle (jerk), which captures how abruptly steering changes occur.
    Lower values indicate smoother, more human-like driving behavior.
    
    Args:
        steering_angles: List of steering angles (radians) at each timestep
        dt: Time step duration (seconds)
        
    Returns:
        Dictionary containing:
            - 'steering_jerk_rms': Root-mean-square of steering jerk (rad/s^3)
            - 'steering_jerk_mean': Mean absolute steering jerk
            - 'steering_jerk_max': Maximum absolute steering jerk
            - 'steering_rate_rms': RMS of first derivative (rad/s)
            - 'steering_accel_rms': RMS of second derivative (rad/s^2)
            
    Formula:
        δ'(t) = dδ/dt (steering rate)
        δ''(t) = d²δ/dt² (steering acceleration)
        δ'''(t) = d³δ/dt³ (steering jerk)
        
    The steering jerk metric captures passenger comfort and actuator wear.
    Lower jerk values indicate smoother control.
    """
    if len(steering_angles) < 4:
        # Need at least 4 points to compute third derivative
        return {
            'steering_jerk_rms': 0.0,
            'steering_jerk_mean': 0.0,
            'steering_jerk_max': 0.0,
            'steering_rate_rms': 0.0,
            'steering_accel_rms': 0.0,
        }
    
    steering_angles = np.array(steering_angles)
    
    # First derivative: steering rate (rad/s)
    steering_rate = np.diff(steering_angles) / dt
    
    # Second derivative: steering acceleration (rad/s^2)
    steering_accel = np.diff(steering_rate) / dt
    
    # Third derivative: steering jerk (rad/s^3)
    steering_jerk = np.diff(steering_accel) / dt
    
    # Compute metrics
    steering_jerk_rms = np.sqrt(np.mean(steering_jerk ** 2))
    steering_jerk_mean = np.mean(np.abs(steering_jerk))
    steering_jerk_max = np.max(np.abs(steering_jerk))
    
    steering_rate_rms = np.sqrt(np.mean(steering_rate ** 2))
    steering_accel_rms = np.sqrt(np.mean(steering_accel ** 2))
    
    return {
        'steering_jerk_rms': float(steering_jerk_rms),
        'steering_jerk_mean': float(steering_jerk_mean),
        'steering_jerk_max': float(steering_jerk_max),
        'steering_rate_rms': float(steering_rate_rms),
        'steering_accel_rms': float(steering_accel_rms),
    }
