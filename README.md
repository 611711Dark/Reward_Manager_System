# Reward Manager System

A flexible priority-based reward management system that allows for hierarchical reward composition with optional logarithmic scaling for output.

## Features

- Priority-based reward hierarchy (higher rank rewards dominate lower ones)
- Variable association for dynamic reward scaling
- Optional logarithmic output scaling
- Clear reward component breakdown
- Easy integration with reinforcement learning environments

## Installation

You can clone the repository directly:

```bash
git clone https://github.com/yourusername/reward-manager.git
cd reward-manager
```

## Basic Usage

### Creating Rewards

```python
from reward_manager import RewardManager

# Create a manager with base 10 (default)
manager = RewardManager()

# Add rewards by rank and parameter
manager.add_reward(rank=2, param=3.0)  # 3.0 * 10^2 = 300
manager.add_reward(rank=1, param=-2.0) # -2.0 * 10^1 = -20

# Or add by direct value (automatically decomposed)
manager.add_value(4500)  # Will be stored as rank=3, param=4.5
```

### Variable Association

```python
speed = 3.0
max_speed = 5.0

# Add speed-dependent reward (scaled by current speed)
manager.add_reward(
    rank=2,
    param=1.0,
    associated_var=speed,
    max_var_value=max_speed,
    multiplier=2.0
)
```

### Reward Calculation

```python
# Get raw total value
raw_total = manager.total_value(use_log=False)

# Get log-scaled total value
log_total = manager.total_value(use_log=True)

print(f"Raw total: {raw_total:.1f}")
print(f"Log total: {log_total:.3f}")
```

### Inspection

```python
# Get highest priority reward
highest = manager.highest_priority()

# Iterate through all rewards
for i, reward in enumerate(manager):
    print(f"Reward {i}: {reward}")

# Clear all rewards
manager.clear()
```

## API Reference

### `PriorityReward` Class

#### `__init__(self, rank: int, param: float, base: int = 10)`
- `rank`: Priority rank (higher = more significant)
- `param`: Reward parameter (can be negative)
- `base`: Numerical base (default 10)

#### Methods:
- `raw_value() -> float`: Returns the raw reward value (param * base^rank)
- `log_value() -> float`: Returns log-scaled value (sign * log10(abs(raw) + 1))

### `RewardManager` Class

#### `__init__(self, base: int = 10)`
- `base`: Numerical base for reward calculation (default 10)

#### Key Methods:

**Adding Rewards:**
- `add_reward(rank, param, associated_var=None, max_var_value=1.0, multiplier=1.0)`
  - Add a reward with optional variable scaling
- `add_value(value, associated_var=None, max_var_value=1.0, multiplier=1.0)`
  - Add reward by value (auto-decomposed to rank/param)

**Calculating Rewards:**
- `total_value(use_log=False) -> float`
  - Calculate sum of all rewards (optionally log-scaled)
  
**Inspection:**
- `highest_priority() -> Optional[PriorityReward]`
- `lowest_priority() -> Optional[PriorityReward]`
- `clear()` - Remove all rewards

## Integration Example

```python
from reward_manager import RewardManager
import numpy as np

class NavigationEnv:
    def __init__(self):
        self.target = np.array([8.0, 8.0])
        self.position = np.array([0.0, 0.0])
        self.speed = 0.0
        self.max_speed = 5.0
        
    def calculate_reward(self):
        manager = RewardManager()
        
        # Distance reward (closer = better)
        distance = np.linalg.norm(self.position - self.target)
        max_dist = np.linalg.norm([10.0, 10.0])  # Max possible distance
        closeness = 1.0 - (distance / max_dist)
        
        manager.add_reward(
            rank=3,
            param=5.0,
            associated_var=closeness,
            max_var_value=1.0,
            multiplier=2.0
        )
        
        # Speed reward
        manager.add_reward(
            rank=2,
            param=1.5,
            associated_var=self.speed,
            max_var_value=self.max_speed
        )
        
        return manager.total_value(use_log=True), manager
```

## Advanced Usage

### Custom Base Values

```python
# Use base 2 for binary-style rewards
binary_manager = RewardManager(base=2)
binary_manager.add_reward(5, 1.0)  # 1.0 * 2^5 = 32
```

### Reward Analysis

```python
# Analyze reward components
total, manager = env.calculate_reward()
print(f"Total reward: {total:.3f}")

print("Breakdown:")
for i, reward in enumerate(manager):
    print(f"{i+1}. Rank {reward.rank}: {reward.param:.2f} = {reward.raw_value():.1f} (log: {reward.log_value():.3f})")
```

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
