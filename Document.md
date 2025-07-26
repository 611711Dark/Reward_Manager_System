# Reward Manager System Design Document

[设计文档](Document_cn.md)

## 1. Core Design Principles

### 1.1 Hierarchical Priority Reward Mechanism

Implemented in the `Reward` class using the core calculation formula:

```python
self._raw = param * (base ** rank)
```

Where:

* `param`: reward parameter, controls the magnitude
* `rank`: priority level, determines the importance of the reward
* `base`: base factor (default is 10), magnifies the difference between levels

This design ensures that higher-rank rewards (e.g., rank=3) naturally dominate lower-rank rewards (e.g., rank=1), without manual weight tuning.

### 1.2 Dynamic Variable Association System

Implemented in the `RewardMgr.add()` method for dynamic adjustment:

```python
if var is not None:
    param *= (var / max_var) * mul
```

Where:

* `var`: current variable value
* `max_var`: maximum value of the variable
* `mul`: multiplier factor (default is 1.0)

## 2. Architectural Design Principles

### 2.1 Logarithmic Output Mode

Log compression implemented in the `Reward.log` property:

```python
sign = -1.0 if self._raw < 0 else 1.0
return sign * math.log(abs(self._raw) + 1, self.base)
```

Design highlights:

* Preserves the original sign
* Applies logarithmic compression to the absolute value
* Adds 1 to prevent log(0) issues

### 2.2 Hierarchical Aggregation Mechanism

Multi-level aggregation in `RewardTrace.to_reward_mgr()`:

```python
for name in all_names:
    total = 0.0
    for rec in self._buf:
        total += rec["named"].get(name, 0.0)
    mgr.add_value(total / n_steps, name=name)
```

This ensures:

* All named reward components are preserved
* Average value of each component is calculated
* A new `RewardMgr` instance is created

### 2.3 Nonlinear Distance Reward

Navigation reward in `simple_env.py`:

```python
distance = np.linalg.norm([self.x - self.target_x, self.y - self.target_y])
max_d = np.linalg.norm([self.max_x, self.max_y])
closeness = 1.0 - (distance / max_d)
mgr.add_value(1000.0, var=closeness ** 0.5, max_var=1.0, mul=2.0, name="distance")
```

Applies square root transformation for nonlinear reward scaling.

## 3. Engineering Highlights

### 3.1 Memory Optimization

Use of `__slots__` to reduce memory usage:

```python
class Reward:
    __slots__ = ("rank", "param", "base", "name", "_raw")
```

### 3.2 Type Safety and Chainable API

Type annotations and fluent API design:

```python
def add(
    self,
    rank: int,
    param: float,
    var: Optional[float] = None,
    max_var: float = 1.0,
    mul: float = 1.0,
    name: Optional[str] = None,
) -> RewardMgr:  # Returns self for chaining
```

### 3.3 Numerical Stability

Handling near-zero values:

```python
if abs(value) < 1e-9:
    rank, param = 0, 0.0
```

## 4. Core Component Details

### 4.1 `Reward` Class

```python
class Reward:
    def __init__(self, rank: int, param: float, base: int = 10, name: Optional[str] = None):
        self.rank = rank
        self.param = param
        self.base = base
        self.name = name
        self._raw = param * (base ** rank)
```

Attributes:

* `raw`: raw reward value
* `log`: log-compressed reward value

### 4.2 `RewardMgr` Class

Main methods:

* `add()`: manually add a reward component
* `add_value()`: automatically decompose a value into rank/param
* `total_raw()`: calculate total raw reward
* `total_log()`: calculate total log reward

### 4.3 `RewardTrace` Class

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

## 5. Example Use Cases

### 5.1 Navigation Environment Integration

Example in `simple_env.py`:

```python
class SimpleNavigationEnv:
    def calculate_reward(self) -> RewardMgr:
        mgr = RewardMgr(base=10)
        mgr.add_value(500.0, name="base")
        mgr.add_value(1000.0, var=self.speed, max_var=self.max_speed, mul=1.5, name="speed")
        # ... other reward components
        return mgr
```

### 5.2 Multi-Level Monitoring System

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

Extending the `RewardTrace` class:

```python
class CustomRewardTrace(RewardTrace):
    def to_reward_mgr(self, mode='avg'):
        if mode == 'max':
            # Max aggregation
        elif mode == 'min':
            # Min aggregation
```

### 6.2 Temporal Decay Mechanism

Apply decay when adding rewards:

```python
decay_factor = math.exp(-0.01 * step_count)
mgr.add(rank, param * decay_factor, name="time_sensitive")
```

### 6.3 Multi-Agent Collaboration

Team reward allocation example:

```python
team_reward = RewardMgr()
for agent in agents:
    individual_reward = agent.calculate_reward()
    team_reward.add_value(individual_reward.total_raw() * 0.7)
team_reward.add_value(global_bonus, name="team_bonus")
```

## 7. Performance Optimization Strategies

### 7.1 High-Frequency Call Optimization

```python
# Disable debug output
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
| `base`    | int   | 10      | Base for level separation; larger values amplify differences |
| `rank`    | int   | -       | Priority level; key indicators recommended to be "5<=rank<=10"         |
| `mul`     | float | 1.5     | Multiplier for dynamic variable-based rewards                |
| `maxlen`  | int   | None    | Maximum length for reward history                            |

Example use cases:

```python
# Safety-critical scenario
mgr = RewardMgr(base=10)
mgr.add(rank=3, param=-2.0, name="collision_penalty")  # High-priority penalty

# Performance optimization
mgr.add(rank=2, param=1.5, var=speed, max_var=max_speed, mul=1.2, name="speed_bonus")
```

