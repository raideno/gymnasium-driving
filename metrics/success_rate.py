import numpy as np

def compute_success_rate(
    episodes_data: list[dict],
    max_lateral_acceleration: float = 4.0,
) -> float:
    """
    Compute the success rate across multiple episodes.
    
    Success is defined as an episode that:
    - Remained within the drivable corridor (on_road = True throughout)
    - Had no collisions (terminated = False due to collision)
    - Had lateral acceleration below max threshold
    
    Args:
        episodes_data: List of episode dictionaries, each containing:
            - 'on_road': List of on_road flags for each timestep
            - 'terminated': Whether episode terminated (collision)
            - 'truncated': Whether episode was truncated (timeout/goal)
            - 'lateral_accelerations': List of lateral accelerations (optional)
        max_lateral_acceleration: Maximum allowed lateral acceleration (m/s^2)
        
    Returns:
        success_rate: Proportion of successful episodes [0, 1]
        
    Formula:
        S_R = (1/N) * Σ I{e_i ∈ E_success}
        
    where I{·} is the indicator function and E_success is the set of episodes
    satisfying all safety constraints.
    """
    if not episodes_data:
        return 0.0
    
    n_episodes = len(episodes_data)
    n_successful = 0
    
    for episode in episodes_data:
        on_road = episode.get('on_road', [])
        terminated = episode.get('terminated', False)
        lateral_accels = episode.get('lateral_accelerations', [])
        
        # Check if stayed on road throughout episode
        stayed_on_road = all(on_road) if on_road else False
        
        # Check if no collision (terminated due to collision, not goal/truncation)
        no_collision = not terminated
        
        # Check if lateral acceleration was within limits
        if lateral_accels:
            lat_accel_ok = all(abs(a) < max_lateral_acceleration for a in lateral_accels)
        else:
            lat_accel_ok = True  # No data means we assume it's ok
        
        # Episode is successful if all conditions are met
        if stayed_on_road and no_collision and lat_accel_ok:
            n_successful += 1
    
    success_rate = n_successful / n_episodes
    return success_rate
