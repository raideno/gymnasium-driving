"""
Wrappers for the bicycle environment.

Provides observation wrappers, reward wrappers, and action wrappers
for customizing the environment for different learning tasks.
"""

from . import actions
from . import rewards
from . import observations

# Re-export commonly used wrappers
from .observations import (
    WithObstaclesInfo,
    WithRoadInfo,
    WithPathInfo,
    WithDynamicsInfo,
)

from .rewards import (
    CombinedReward,
    SimpleRewardWrapper
)

from .actions import (
    DiscreteActionWrapper,
    MultiDiscreteActionWrapper,
    CoarseDiscreteActionWrapper,
    FlattenMultiDiscreteWrapper
)

__all__ = [
    # Modules
    "observations",
    "rewards",
    "actions",
    
    # Observation wrappers
    "WithObstaclesInfo",
    "WithRoadInfo",
    "WithPathInfo",
    "WithDynamicsInfo",
    
    # Reward wrappers
    "CombinedReward",
    "SimpleRewardWrapper",
    
    # Actions
    "DiscreteActionWrapper",
    "MultiDiscreteActionWrapper",
    "CoarseDiscreteActionWrapper",
    "FlattenMultiDiscreteWrapper"
]
