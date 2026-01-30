import numpy as np

def compute_cross_track_error(
    positions: list[np.ndarray],
    reference_path: np.ndarray,
) -> dict[str, float]:
    """
    Compute the Cross-Track Error (CTE) metrics for a trajectory.
    
    CTE measures the minimum lateral deviation of the vehicle's position
    from the reference path at each timestep.
    
    Args:
        positions: List of vehicle positions (x, y) at each timestep
        reference_path: Reference path waypoints as (N, 2) array
        
    Returns:
        Dictionary containing:
            - 'cte_rms': Root-mean-square CTE over the episode
            - 'cte_mean': Mean CTE
            - 'cte_max': Maximum CTE
            - 'cte_std': Standard deviation of CTE
            
    Formula:
        CTE(t) = min_{w ∈ P_ref} ||p_cog(t) - w||_2
        
        CTE_RMS = sqrt((1/T) * Σ CTE(t)^2)
        
    where p_cog(t) is the vehicle's center of gravity position at time t,
    and P_ref is the reference path.
    
    Lower CTE_RMS values indicate superior path tracking accuracy.
    """
    if not positions or len(reference_path) == 0:
        return {
            'cte_rms': float('inf'),
            'cte_mean': float('inf'),
            'cte_max': float('inf'),
            'cte_std': 0.0,
        }
    
    positions = np.array(positions)
    cte_values = []
    
    # Compute CTE for each position
    for pos in positions:
        # Calculate distance to all waypoints
        distances = np.linalg.norm(reference_path - pos, axis=1)
        # Minimum distance is the CTE
        min_distance = np.min(distances)
        cte_values.append(min_distance)
    
    cte_values = np.array(cte_values)
    
    # Compute statistics
    cte_rms = np.sqrt(np.mean(cte_values ** 2))
    cte_mean = np.mean(cte_values)
    cte_max = np.max(cte_values)
    cte_std = np.std(cte_values)
    
    return {
        'cte_rms': float(cte_rms),
        'cte_mean': float(cte_mean),
        'cte_max': float(cte_max),
        'cte_std': float(cte_std),
    }
