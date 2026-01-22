# Reward Manager System (RMS)

[Chinese Version](README_cn.md) | [Full Design Document](Document.md)

## Project Overview

Reward Manager System (RMS) is a straightforward reward management system designed for reinforcement learning and complex decision-making systems. It provides **direct reward value control**, **dynamic variable association**, **reward clipping**, **rich visualization**, and **curriculum learning** support.

The core design is simple and intuitive: **directly input reward values** (like 0.1, 0.04, 5.0, etc.) and optionally scale them based on dynamic variables.

## Key Features

1. **Direct Value Control**
   - Input reward values directly (e.g., 0.1, 0.04, 5.0)
   - No complex rank/param configuration needed
   - Intuitive and easy to tune

2. **Dynamic Variable Association**

   ```python
   # Speed reward: dynamically adjusted based on current speed
   mgr.add(1.0, var=current_speed, max_var=max_speed, mul=1.5, name="speed")
   ```

3. **Reward Clipping**
   - Prevent extreme reward values
   - Supports both range and single-value clipping

   ```python
   # Range clipping [0, 5]
   mgr.add(10.0, name="bonus", clip=(0, 5))

   # Upper limit only
   mgr.add(10.0, name="reward", clip=5)
   ```

4. **Curriculum Learning**
   - Multi-stage reward progression
   - Support for episode-based, game-based, and performance-based triggers

   ```python
   stage1 = Stage("easy", episodes=100)
   stage2 = Stage("medium", condition=lambda: success_rate > 0.8)
   ```

5. **Rich Visualization**
   - Heatmap for reward components over time
   - Correlation matrix for component relationships
   - Distribution histograms for each component
   - Comprehensive dashboard

   ```python
   trace.plot_heatmap(save_path="heatmap.png")
   trace.plot_correlation(save_path="correlation.png")
   trace.plot_distribution(save_path="distribution.png")
   trace.plot_dashboard(save_path="dashboard.png")
   ```

6. **Multi-level Aggregation and Compression**

   ```mermaid
   graph TD
     A[Step-Level Reward] -->|50 steps| B[Game-Level Aggregation]
     B -->|50 games| C[Episode-Level Aggregation]
     C -->|60 episodes| D[Training Analysis]
   ```

7. **Dual Output Modes**

   * `raw`: raw reward value (preserves magnitude differences)
   * `log`: log-compressed value (suitable for neural network training)

## Installation and Usage

### Installation

```bash
git clone https://github.com/611711Dark/Reward_Manager_System.git
pip install numpy matplotlib  # for visualization features
```

### Basic Usage

```python
from reward_system import RewardMgr

# Create a reward manager
mgr = RewardMgr()

# Add a fixed base reward
mgr.add(5.0, name="base")

# Add a dynamic speed reward with clipping
mgr.add(3.0, var=5.0, max_var=10.0, mul=1.5, name="speed", clip=(0, 5))

print(f"Raw Reward: {mgr.total_raw():.3f}")  # Raw Reward: 5.000
print(f"Log Reward: {mgr.total_log():.3f}")  # Log Reward: 0.699
print(f"Speed Component: {mgr['speed']:.3f}")  # Speed Component: 2.250
```

### Curriculum Learning Example

```python
from reward_system import CurriculumMgr, Stage

# Define training stages
curriculum = CurriculumMgr()

# Stage 1: Basic movement (episodes 0-100)
stage1 = Stage("easy", episodes=100)
stage1.add(1.0, name="reach_target", clip=(0, 1))
stage1.add(0.5, name="not_crash", clip=(0, 1))

# Stage 2: Speed control (triggered when success rate > 0.8)
stage2 = Stage("medium", condition=lambda: success_rate > 0.8)
stage2.add(1.0, name="reach_target", clip=(0, 1))
stage2.add(0.8, name="speed_bonus", var=speed/max_speed)
stage2.add(0.5, name="not_crash")

# Stage 3: Full task (final stage)
stage3 = Stage("hard")
stage3.add(1.0, name="reach_target")
stage3.add(0.8, name="speed_bonus")
stage3.add(0.5, name="efficiency")

curriculum.add_stages(stage1, stage2, stage3)

# Training loop
for ep in range(500):
    # Get current stage reward
    mgr = curriculum.get_reward()

    # Execute action...
    state, reward, done = env.step(action)

    # Update progress and check for stage advancement
    if curriculum.advance(episode_count=ep):
        print(f"Advanced to: {curriculum.get_current_stage().name}")
```

### Visualization Example

```python
from reward_system import RewardTrace

# Record rewards during training
trace = RewardTrace()
for step in range(100):
    mgr = env.calculate_reward()
    trace.push(mgr)

# Visualize
trace.plot_dashboard(save_path="dashboard.png")
trace.plot_heatmap(save_path="heatmap.png")
trace.plot_correlation(save_path="correlation.png")
trace.plot_distribution(save_path="distribution.png")
```

## Core Components

### 1. Reward (Atomic Reward)

```python
r = Reward(0.5, name="critical")
print(r.raw)  # 0.5
print(r.log)  # 0.17609125905568124
```

### 2. RewardMgr (Reward Manager)

```python
mgr = RewardMgr()
mgr.add(2.0, name="bonus", clip=(0, 5))  # With clipping
mgr.add(-1.0, name="penalty", clip=(-10))  # Upper limit only

# Chain calls
mgr.add(5.0, name="base").add(-0.5, name="error")
```

### 3. Stage (Training Stage)

```python
# Episode-based trigger
stage = Stage("easy", episodes=100)

# Performance-based trigger
stage = Stage("medium", condition=lambda: success_rate > 0.8)

# No trigger (final stage)
stage = Stage("hard")

# Add rewards to stage
stage.add(1.0, name="reward", clip=(0, 2))
```

### 4. CurriculumMgr (Curriculum Manager)

```python
curriculum = CurriculumMgr()
curriculum.add_stage(stage1).add_stage(stage2).add_stage(stage3)

# Or批量添加
curriculum.add_stages(stage1, stage2, stage3)

# Get current reward
mgr = curriculum.get_reward()

# Check and advance stages
if curriculum.advance(episode_count=150):
    print("Stage advanced!")
```

### 5. RewardTrace (Reward Trace)

```python
trace = RewardTrace()

# Record multi-step rewards
for _ in range(10):
    mgr = env.calculate_reward()
    trace.push(mgr)

# Compress into a single RewardMgr
summary = trace.to_reward_mgr()

# Visualize
trace.plot_heatmap()
trace.plot_correlation()
trace.plot_distribution()
trace.plot_dashboard()
```

## API Reference

### RewardMgr.add()

```python
def add(
    self,
    value: float,
    var: Optional[float] = None,
    max_var: float = 1.0,
    mul: float = 1.0,
    name: Optional[str] = None,
    clip: Optional[Union[float, tuple[float, float]]] = None,
) -> RewardMgr
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value` | float | - | Direct reward value (e.g., 0.1, 0.04) |
| `var` | float | None | Dynamic variable value (optional) |
| `max_var` | float | 1.0 | Max value of the variable for normalization |
| `mul` | float | 1.0 | Multiplier factor |
| `name` | str | None | Reward name for querying |
| `clip` | float/tuple | None | Clipping limit: `max` or `(min, max)` |

### Stage()

```python
def __init__(
    self,
    name: str,
    episodes: Optional[int] = None,
    games: Optional[int] = None,
    condition: Optional[Callable[[], bool]] = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | - | Stage name |
| `episodes` | int | None | Trigger after this many episodes |
| `games` | int | None | Trigger after this many games |
| `condition` | callable | None | Custom condition function |

### CurriculumMgr.advance()

```python
def advance(
    self,
    episode_count: Optional[int] = None,
    game_count: Optional[int] = None,
    check_condition: bool = True,
) -> bool
```

Returns `True` if stage was advanced, `False` otherwise.

### RewardTrace Visualization Methods

| Method | Description |
|---------|-------------|
| `plot_heatmap(save_path=None, title="...")` | Reward heatmap over time |
| `plot_correlation(save_path=None, title="...")` | Component correlation matrix |
| `plot_distribution(save_path=None, title="...")` | Distribution histograms |
| `plot_dashboard(save_path=None)` | Comprehensive dashboard |

## Application Scenarios

1. **Reinforcement Learning Systems**

   * Replace traditional scalar rewards
   * Address sparse reward issues
   * Use curriculum learning for progressive difficulty

2. **Game AI Development**

   * Compose complex behavior rewards
   * Balance multiple objectives
   * Progressively unlock advanced mechanics

3. **Robot Control**

   * Prioritize safety constraints
   * Fuse multi-sensor reward signals
   * Start with basic tasks, advance to complex ones

## Demo Files

| File | Description |
|------|-------------|
| `demo.py` | Three-level monitoring demo |
| `curriculum_demo.py` | Curriculum learning demo |
| `simple_env.py` | Simple navigation environment |

Run demos:

```bash
python demo.py
python curriculum_demo.py
```

## Contribution Guide

We welcome contributions via issues or pull requests:

1. Report bugs or suggestions
2. Add new environment examples
3. Extend visualization features
4. Optimize core algorithms

## License

This project is licensed under the [MIT License](LICENSE).

---
