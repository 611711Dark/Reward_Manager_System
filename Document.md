# Reward Manager System Design Document

[中文版 (Chinese Version)](Document_cn.md)

## 1. Core Design Principles

### 1.1 Direct Value Reward Mechanism

Implemented in the `Reward` class with a straightforward approach:

```python
self.value = value
```

Where:
* `value`: direct reward value (e.g., 0.1, 0.04, 5.0)

This design ensures simplicity and intuitiveness - users directly input reward values without needing to understand complex rank/param decomposition.

### 1.2 Dynamic Variable Association System

Implemented in the `RewardMgr.add()` method for dynamic adjustment:

```python
if var is not None:
    value = value * (var / max_var) * mul
```

Where:
* `var`: current variable value
* `max_var`: maximum value of the variable
* `mul`: multiplier factor (default is 1.0)

### 1.3 Reward Clipping Mechanism

Implemented in the `RewardMgr.add()` method to prevent extreme values:

```python
if clip is not None:
    if isinstance(clip, tuple):
        min_val, max_val = clip
        if min_val is not None:
            value = max(value, min_val)
        if max_val is not None:
            value = min(value, max_val)
    else:  # single value means upper limit
        value = min(value, clip)
```

This provides safety against outliers and unexpected large reward values.

## 2. Architectural Design Principles

### 2.1 Logarithmic Output Mode

Log compression implemented in the `Reward.log` property:

```python
if abs(self.value) < 1e-9:
    return 0.0
sign = -1.0 if self.value < 0 else 1.0
return sign * math.log(abs(self.value) + 1, 10)
```

Design highlights:
* Preserves the original sign
* Applies logarithmic compression to the absolute value
* Adds 1 to prevent log(0) issues
* Handles near-zero values appropriately

### 2.2 Multi-level Aggregation Mechanism

Multi-level aggregation in `RewardTrace.to_reward_mgr()`:

```python
for name in all_names:
    total = 0.0
    for rec in self._buf:
        total += rec["named"].get(name, 0.0)
    mgr.add(total / n_steps, name=name)
```

This ensures:
* All named reward components are preserved
* Average value of each component is calculated
* A new `RewardMgr` instance is created

### 2.3 Curriculum Learning Architecture

Implemented in `Stage` and `CurriculumMgr` classes:

```mermaid
graph TD
    A[Stage 1: Easy] -->|condition met| B[Stage 2: Medium]
    B -->|condition met| C[Stage 3: Hard]

    D[CurriculumMgr] --> E[advance()]
    E -->|check conditions| F{All conditions met?}
    F -->|Yes| G[Switch to next stage]
    F -->|No| H[Stay in current stage]
```

**Stage triggers:**
1. Episode-based: `Stage("easy", episodes=100)`
2. Game-based: `Stage("medium", games=500)`
3. Performance-based: `Stage("hard", condition=lambda: success_rate > 0.8)`
4. Combined: `Stage("advanced", episodes=100, condition=lambda: reward > 10)`

## 3. Engineering Highlights

### 3.1 Memory Optimization

Use of `__slots__` to reduce memory usage:

```python
class Reward:
    __slots__ = ("value", "name")
```

### 3.2 Type Safety and Chainable API

Type annotations and fluent API design:

```python
def add(
    self,
    value: float,
    var: Optional[float] = None,
    max_var: float = 1.0,
    mul: float = 1.0,
    name: Optional[str] = None,
    clip: Optional[Union[float, tuple[float, float]]] = None,
) -> RewardMgr:  # Returns self for chaining
```

### 3.3 Numerical Stability

Handling near-zero values:

```python
if abs(self.value) < 1e-9:
    return 0.0
```

### 3.4 Visualization System

Implemented in `RewardTrace` class with lazy imports:

```python
def plot_heatmap(self, save_path=None, title="Reward Heatmap"):
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Visualization requires numpy and matplotlib")
```

This allows optional visualization without requiring dependencies for core functionality.

## 4. Core Component Details

### 4.1 `Reward` Class

```python
class Reward:
    def __init__(self, value: float, name: Optional[str] = None):
        self.value = value
        self.name = name
```

Properties:
* `raw`: raw reward value (same as `value`)
* `log`: log-compressed reward value (base 10)

### 4.2 `RewardMgr` Class

Main methods:
* `add()`: add a reward component with optional dynamic variable scaling and clipping
* `total_raw()`: calculate total raw reward
* `total_log()`: calculate total log reward

### 4.3 `Stage` Class

Training stage definition:

```python
class Stage:
    def __init__(
        self,
        name: str,
        episodes: Optional[int] = None,
        games: Optional[int] = None,
        condition: Optional[Callable[[], bool]] = None,
    ):
```

Key methods:
* `add()`: add reward components to this stage
* `get_reward()`: get a fresh `RewardMgr` with this stage's rewards

### 4.4 `CurriculumMgr` Class

Curriculum learning manager:

```python
class CurriculumMgr:
    def add_stage(self, stage: Stage) -> "CurriculumMgr"
    def add_stages(self, *stages: Stage) -> "CurriculumMgr"
    def get_current_stage(self) -> Optional[Stage]
    def get_reward(self) -> RewardMgr
    def advance(...) -> bool  # Returns True if stage advanced
```

### 4.5 `RewardTrace` Class

Reward trace recorder:

```python
class RewardTrace:
    def __init__(self, maxlen: Optional[int] = None):
        self._buf = deque(maxlen=maxlen)

    def push(self, mgr: RewardMgr) -> RewardTrace:
        # Record reward snapshot
```

Key methods:
* `arrays()`: convert trace to dictionary of arrays
* `to_reward_mgr()`: aggregate trace into a single `RewardMgr`
* `plot_heatmap()`: visualize rewards as heatmap
* `plot_correlation()`: visualize component correlations
* `plot_distribution()`: visualize component distributions
* `plot_dashboard()`: comprehensive visualization dashboard

## 5. Example Use Cases

### 5.1 Reward Clipping Examples

```python
# Prevent extreme positive rewards
mgr.add(100.0, name="bonus", clip=(0, 10))

# Only limit upper bound
mgr.add(-50.0, name="penalty", clip=(-10))

# Clip both directions
mgr.add(5.0, name="reward", clip=(-2, 2))
```

### 5.2 Navigation Environment Integration

Example in `simple_env.py`:

```python
class SimpleNavigationEnv:
    def calculate_reward(self) -> RewardMgr:
        mgr = RewardMgr()

        mgr.add(5.0, name="base")

        mgr.add(3.0, var=self.speed, max_var=self.max_speed,
                mul=1.5, name="speed", clip=(0, 5))

        mgr.add(-2.0, var=abs(self.direction_error),
                max_var=30, mul=2.0, name="direction")

        distance = np.linalg.norm([self.x - self.target_x, self.y - self.target_y])
        max_d = np.linalg.norm([self.max_x, self.max_y])
        closeness = 1.0 - (distance / max_d)
        mgr.add(2.0, var=closeness ** 0.5, max_var=1.0,
                mul=2.0, name="distance")

        return mgr
```

### 5.3 Curriculum Learning Training

```python
# Define stages
curriculum = CurriculumMgr()

# Stage 1: Basic movement
stage1 = Stage("easy", episodes=100)
stage1.add(1.0, name="reach_target", clip=(0, 1))

# Stage 2: Performance-based advancement
stage2 = Stage("medium", condition=lambda: success_rate > 0.8)
stage2.add(1.0, name="reach_target")
stage2.add(0.8, name="speed_bonus", clip=(0, 1))

# Stage 3: Full task
stage3 = Stage("hard")
stage3.add(1.0, name="reach_target")
stage3.add(0.8, name="speed_bonus")
stage3.add(0.5, name="efficiency", clip=(0, 1))

curriculum.add_stages(stage1, stage2, stage3)

# Training loop
for ep in range(500):
    mgr = curriculum.get_reward()
    state, reward, done = env.step(action)

    # Check for stage advancement
    if curriculum.advance(episode_count=ep):
        print(f"Stage advanced: {curriculum.get_current_stage().name}")
```

### 5.4 Multi-Level Monitoring

Three-level monitoring in `demo.py`:

```python
# Step-level monitoring
step_arrays = final_step_trace.arrays()
axes[0].plot(step_arrays["raw"], label="raw")

# Game-level monitoring
game_arrays = final_game_trace.arrays()
axes[1].plot(game_arrays["log"], label="log")

# Episode-level monitoring
ep_arrays = episode_hist.arrays()
axes[2].plot(ep_arrays["distance"], label="distance")
```

## 6. Extensions and Customization

### 6.1 Custom Aggregation Strategy

Extending `RewardTrace` class:

```python
class CustomRewardTrace(RewardTrace):
    def to_reward_mgr(self, mode='avg'):
        if mode == 'max':
            # Max aggregation
        elif mode == 'min':
            # Min aggregation
```

### 6.2 Custom Stage Conditions

Using complex conditions:

```python
# Multi-factor condition
stage = Stage("advanced",
    condition=lambda: success_rate > 0.8 and avg_reward > 5.0)

# History-based condition
stage = Stage("master",
    condition=lambda: len(recent_wins) >= 5)

# Time-based condition
stage = Stage("timed",
    condition=lambda: time_elapsed > 3600)
```

### 6.3 Visualization Customization

Custom visualization:

```python
def plot_custom_metrics(trace):
    data = trace.arrays()
    # Custom plotting logic
    ...
```

## 7. Performance Optimization Strategies

### 7.1 High-Frequency Call Optimization

```python
# Disable debug output for production
if not DEBUG_MODE:
    Reward.__repr__ = lambda self: ""
```

### 7.2 Large-Scale Data Handling

```python
# Use sliding window to limit history
reward_trace = RewardTrace(maxlen=1000)  # Keep only the most recent 1000 records
```

### 7.3 Serialization Optimization

```python
# Custom serialization methods
class RewardTrace:
    def serialize(self):
        return list(self._buf)

    @classmethod
    def deserialize(cls, data):
        trace = cls()
        trace._buf = deque(data)
        return trace
```

## 8. Parameter Tuning Guide

| Parameter | Type  | Default | Description                                                  |
| --------- | ----- | ------- | ------------------------------------------------------------ |
| `value`   | float | -       | Direct reward value (e.g., 0.1, 0.04, 5.0)                    |
| `var`     | float | None    | Dynamic variable value for scaling                          |
| `max_var` | float | 1.0     | Maximum value of the variable for normalization             |
| `mul`     | float | 1.0     | Multiplier factor for dynamic variable-based rewards         |
| `maxlen`  | int   | None    | Maximum length for reward history                            |
| `clip`    | float/tuple | None | Clipping limit: single value (upper) or (min, max) |
| `episodes`| int   | None    | Trigger stage after this many episodes                |
| `games`   | int   | None    | Trigger stage after this many games                    |
| `condition`| callable | None    | Custom function that returns True to trigger stage      |

### Example Use Cases

```python
# Safety-critical scenario - use larger penalty values with clipping
mgr = RewardMgr()
mgr.add(-10.0, name="collision_penalty", clip=(-5))  # Limit to -5 max

# Performance optimization - use moderate positive values
mgr.add(1.5, var=speed, max_var=max_speed, mul=1.2, name="speed_bonus")

# Small nudges - use very small values
mgr.add(0.1, name="small_correction")
mgr.add(0.04, name="fine_tune")
```

### Reward Value Scales

| Scale | Range | Use Case |
|-------|-------|----------|
| Large | > 1.0 | Major achievements, critical penalties |
| Medium | 0.1 - 1.0 | Good performance, moderate penalties |
| Small | 0.01 - 0.1 | Minor improvements, small corrections |
| Fine | < 0.01 | Fine-tuning, nudges |

### Curriculum Learning Best Practices

1. **Start Simple**: Begin with 1-2 reward components
2. **Progressive Addition**: Add more components as agent improves
3. **Performance-Based Triggers**: Prefer performance over fixed episode counts
4. **Clear Separation**: Ensure stages have distinct reward structures
5. **Monitor Transitions**: Log and visualize stage switches

## 9. Visualization Features

### 9.1 Heatmap Visualization

Shows reward component values over time as a color-coded grid:

```python
trace.plot_heatmap(save_path="heatmap.png", cmap="RdYlGn")
```

**Use cases:**
- Identify periods of high/low reward for each component
- Spot correlations between components
- Visualize stage transitions

### 9.2 Correlation Matrix

Shows correlation coefficients between all reward components:

```python
trace.plot_correlation(save_path="correlation.png")
```

**Use cases:**
- Find redundant reward components
- Identify conflicting rewards (negative correlation)
- Validate reward design

### 9.3 Distribution Visualization

Shows histogram of each reward component:

```python
trace.plot_distribution(save_path="distribution.png")
```

**Use cases:**
- Check reward distribution shape
- Identify skew or outliers
- Verify clipping effectiveness

### 9.4 Dashboard Visualization

Comprehensive view combining all visualization types:

```python
trace.plot_dashboard(save_path="dashboard.png")
```

Includes:
- Reward heatmap
- Correlation matrix
- Total reward trends
- Individual component trends

## 10. Curriculum Learning Design Patterns

### 10.1 Episode-Based Progression

```python
Stage("easy", episodes=100)
Stage("medium", episodes=200)
Stage("hard")  # No limit
```

**Best for**: Fixed-length training, predictable progression

### 10.2 Performance-Based Progression

```python
Stage("medium", condition=lambda: success_rate > 0.7)
Stage("hard", condition=lambda: avg_reward > 5.0)
```

**Best for**: Adaptive training, quality-focused progression

### 10.3 Hybrid Progression

```python
Stage("intermediate",
    episodes=100,  # Minimum
    condition=lambda: success_rate > 0.5)  # Quality gate
```

**Best for**: Balanced approach, ensures minimum exposure

### 10.4 Component Gradation

Gradually add reward components across stages:

```python
# Stage 1: Base only
stage1.add(1.0, name="reach_target")

# Stage 2: Add speed
stage2.add(1.0, name="reach_target")
stage2.add(0.5, name="speed_bonus")

# Stage 3: Add efficiency
stage3.add(1.0, name="reach_target")
stage3.add(0.5, name="speed_bonus")
stage3.add(0.3, name="efficiency")
```

**Best for**: Teaching agent multiple skills progressively
