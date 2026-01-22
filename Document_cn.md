# Reward Manager 系统设计文档

[Full Design Document](Document.md)

## 1. 核心设计思想

### 1.1 直接值奖励机制

在 `Reward` 类中实现的核心设计：
```python
self.value = value
```
其中：
- `value`：直接奖励值（如 0.1, 0.04, 5.0）

这种设计确保了简单直观 - 用户直接输入奖励值，无需理解复杂的 rank/param 分解。

### 1.2 动态变量关联系统

在 `RewardMgr.add()` 方法中实现的动态调节：
```python
if var is not None:
    value = value * (var / max_var) * mul
```
其中：
- `var`：当前变量值
- `max_var`：变量最大值
- `mul`：乘数因子（默认 1.0）

### 1.3 奖励裁剪机制

在 `RewardMgr.add()` 方法中实现，防止极端值：
```python
if clip is not None:
    if isinstance(clip, tuple):
        min_val, max_val = clip
        if min_val is not None:
            value = max(value, min_val)
        if max_val is not None:
            value = min(value, max_val)
    else:  # 单个值表示上限
        value = min(value, clip)
```

这提供了对异常值和意外大奖励值的保护。

## 2. 架构设计原理

### 2.1 对数输出模式

在 `Reward.log` 属性中实现的对数压缩：
```python
if abs(self.value) < 1e-9:
    return 0.0
sign = -1.0 if self.value < 0 else 1.0
return sign * math.log(abs(self.value) + 1, 10)
```
这种设计：
- 保持原始值的符号
- 对绝对值取对数压缩
- 加 1 防止对 0 取对数
- 适当处理接近零的值

### 2.2 多级聚合机制

在 `RewardTrace.to_reward_mgr()` 中实现的多级聚合：
```python
for name in all_names:
    total = 0.0
    for rec in self._buf:
        total += rec["named"].get(name, 0.0)
    mgr.add(total / n_steps, name=name)
```
这种设计：
- 保留所有命名奖励组件
- 计算每个组件的平均值
- 创建新的 RewardMgr 实例

### 2.3 课程学习架构

在 `Stage` 和 `CurriculumMgr` 类中实现：

```mermaid
graph TD
    A[阶段1: 简单] -->|条件满足| B[阶段2: 中等]
    B -->|条件满足| C[阶段3: 困难]

    D[CurriculumMgr] --> E[advance()]
    E -->|检查条件| F{所有条件都满足?}
    F -->|是| G[切换到下一阶段]
    F -->|否| H[保持在当前阶段]
```

**阶段触发方式：**
1. 基于 episode 数量：`Stage("easy", episodes=100)`
2. 基于 game 数量：`Stage("medium", games=500)`
3. 基于性能指标：`Stage("hard", condition=lambda: success_rate > 0.8)`
4. 组合触发：`Stage("advanced", episodes=100, condition=lambda: reward > 10)`

## 3. 工程实现特色

### 3.1 内存优化

使用 `__slots__` 减少内存占用：
```python
class Reward:
    __slots__ = ("value", "name")
```

### 3.2 类型安全与链式 API

类型注解和链式调用设计：
```python
def add(
    self,
    value: float,
    var: Optional[float] = None,
    max_var: float = 1.0,
    mul: float = 1.0,
    name: Optional[str] = None,
    clip: Optional[Union[float, tuple[float, float]]] = None,
) -> RewardMgr:  # 返回自身类型，支持链式调用
```

### 3.3 数值稳定性处理

处理极小值的保护机制：
```python
if abs(self.value) < 1e-9:
    return 0.0
```

### 3.4 可视化系统

在 `RewardTrace` 类中实现，使用惰性导入：
```python
def plot_heatmap(self, save_path=None, title="奖励热图"):
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("可视化需要安装 numpy 和 matplotlib")
```

这使得可视化功能可选，核心功能不需要额外依赖。

## 4. 核心组件详解

### 4.1 Reward 类

```python
class Reward:
    def __init__(self, value: float, name: Optional[str] = None):
        self.value = value
        self.name = name
```

属性：
- `raw`: 原始奖励值（与 `value` 相同）
- `log`: 对数压缩后的奖励值（以 10 为底）

### 4.2 RewardMgr 类

主要方法：
- `add()`: 添加奖励组件，支持动态变量缩放和裁剪
- `total_raw()`: 计算原始奖励总和
- `total_log()`: 计算对数奖励总和

### 4.3 Stage 类

训练阶段定义：
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

关键方法：
- `add()`: 向该阶段添加奖励组件
- `get_reward()`: 获取该阶段的新 RewardMgr

### 4.4 CurriculumMgr 类

课程学习管理器：
```python
class CurriculumMgr:
    def add_stage(self, stage: Stage) -> "CurriculumMgr"
    def add_stages(self, *stages: Stage) -> "CurriculumMgr"
    def get_current_stage(self) -> Optional[Stage]
    def get_reward(self) -> RewardMgr
    def advance(...) -> bool  # 返回 True 表示阶段已推进
```

### 4.5 RewardTrace 类

奖励轨迹记录器：
```python
class RewardTrace:
    def __init__(self, maxlen: Optional[int] = None):
        self._buf = deque(maxlen=maxlen)

    def push(self, mgr: RewardMgr) -> RewardTrace:
        # 记录奖励快照
```

关键方法：
- `arrays()`: 将轨迹转换为字典数组
- `to_reward_mgr()`: 聚合轨迹为单个 RewardMgr
- `plot_heatmap()`: 可视化奖励热图
- `plot_correlation()`: 可视化组件相关性
- `plot_distribution()`: 可视化组件分布
- `plot_dashboard()`: 综合可视化仪表板

## 5. 应用场景示例

### 5.1 奖励裁剪示例

```python
# 防止极端正向奖励
mgr.add(100.0, name="bonus", clip=(0, 10))

# 只限制上限
mgr.add(-50.0, name="penalty", clip=(-10))

# 双向裁剪
mgr.add(5.0, name="reward", clip=(-2, 2))
```

### 5.2 导航环境集成

在 `simple_env.py` 中的实现：
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

### 5.3 课程学习训练

```python
# 定义阶段
curriculum = CurriculumMgr()

# 阶段1：基础移动
stage1 = Stage("easy", episodes=100)
stage1.add(1.0, name="reach_target", clip=(0, 1))

# 阶段2：基于性能的推进
stage2 = Stage("medium", condition=lambda: success_rate > 0.8)
stage2.add(1.0, name="reach_target")
stage2.add(0.8, name="speed_bonus", clip=(0, 1))

# 阶段3：完整任务
stage3 = Stage("hard")
stage3.add(1.0, name="reach_target")
stage3.add(0.8, name="speed_bonus")
stage3.add(0.5, name="efficiency", clip=(0, 1))

curriculum.add_stages(stage1, stage2, stage3)

# 训练循环
for ep in range(500):
    mgr = curriculum.get_reward()
    state, reward, done = env.step(action)

    # 检查阶段推进
    if curriculum.advance(episode_count=ep):
        print(f"阶段已推进: {curriculum.get_current_stage().name}")
```

### 5.4 多级监控

在 `demo.py` 中实现的三级监控：
```python
# Step 级监控
step_arrays = final_step_trace.arrays()
axes[0].plot(step_arrays["raw"], label="raw")

# Game 级监控
game_arrays = final_game_trace.arrays()
axes[1].plot(game_arrays["log"], label="log")

# Episode 级监控
ep_arrays = episode_hist.arrays()
axes[2].plot(ep_arrays["distance"], label="distance")
```

## 6. 扩展与定制

### 6.1 自定义聚合策略

扩展 `RewardTrace` 类：
```python
class CustomRewardTrace(RewardTrace):
    def to_reward_mgr(self, mode='avg'):
        if mode == 'max':
            # 最大值聚合
        elif mode == 'min':
            # 最小值聚合
```

### 6.2 自定义阶段条件

使用复杂条件：
```python
# 多因子条件
stage = Stage("advanced",
    condition=lambda: success_rate > 0.8 and avg_reward > 5.0)

# 基于历史的条件
stage = Stage("master",
    condition=lambda: len(recent_wins) >= 5)

# 基于时间的条件
stage = Stage("timed",
    condition=lambda: time_elapsed > 3600)
```

### 6.3 可视化定制

自定义可视化：
```python
def plot_custom_metrics(trace):
    data = trace.arrays()
    # 自定义绘图逻辑
    ...
```

## 7. 性能优化策略

### 7.1 高频调用优化
```python
# 禁用调试输出
if not DEBUG_MODE:
    Reward.__repr__ = lambda self: ""
```

### 7.2 大规模数据处理
```python
# 使用滑动窗口限制历史数据
reward_trace = RewardTrace(maxlen=1000)  # 只保留最近的 1000 条记录
```

### 7.3 序列化优化
```python
# 自定义序列化方法
class RewardTrace:
    def serialize(self):
        return list(self._buf)

    @classmethod
    def deserialize(cls, data):
        trace = cls()
        trace._buf = deque(data)
        return trace
```

## 8. 参数调优指南

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `value` | float | - | 直接奖励值（如 0.1, 0.04, 5.0） |
| `var` | float | None | 用于缩放的动态变量值 |
| `max_var` | float | 1.0 | 变量最大值，用于归一化 |
| `mul` | float | 1.0 | 动态变量奖励的倍数因子 |
| `maxlen` | int | None | 历史记录最大长度 |
| `clip` | float/元组 | None | 裁剪限制：单个值（上限）或 (min, max) |
| `episodes` | int | None | 达到此 episode 数量后触发阶段 |
| `games` | int | None | 达到此 game 数量后触发阶段 |
| `condition` | callable | None | 返回 True 时触发阶段的自定义函数 |

实际应用示例：
```python
# 安全关键型应用 - 使用较大的惩罚值并裁剪
mgr = RewardMgr()
mgr.add(-10.0, name="collision_penalty", clip=(-5))  # 限制最大 -5

# 性能优化应用 - 使用适中的正值
mgr.add(1.5, var=speed, max_var=max_speed, mul=1.2, name="speed_bonus")

# 微调 - 使用非常小的值
mgr.add(0.1, name="small_correction")
mgr.add(0.04, name="fine_tune")
```

### 奖励值规模参考

| 规模 | 范围 | 使用场景 |
|------|------|----------|
| 大 | > 1.0 | 重大成就、关键惩罚 |
| 中 | 0.1 - 1.0 | 良好表现、中等惩罚 |
| 小 | 0.01 - 0.1 | 小幅改进、小修正 |
| 微调 | < 0.01 | 精细调整、微调 |

### 课程学习最佳实践

1. **从简单开始**：先用 1-2 个奖励组件
2. **渐进式添加**：随着智能体改进添加更多组件
3. **基于性能的触发**：优先使用性能指标而非固定的 episode 数量
4. **清晰的分离**：确保各阶段有不同的奖励结构
5. **监控转换**：记录并可视化阶段切换

## 9. 可视化功能

### 9.1 热图可视化

以颜色编码网格形式显示奖励组件随时间的变化：

```python
trace.plot_heatmap(save_path="heatmap.png", cmap="RdYlGn")
```

**使用场景：**
- 识别各组件的高/低奖励时期
- 发现组件间的相关性
- 可视化阶段转换

### 9.2 相关性矩阵

显示所有奖励组件之间的相关系数：

```python
trace.plot_correlation(save_path="correlation.png")
```

**使用场景：**
- 发现冗余的奖励组件
- 识别冲突的奖励（负相关）
- 验证奖励设计

### 9.3 分布可视化

显示每个奖励组件的直方图：

```python
trace.plot_distribution(save_path="distribution.png")
```

**使用场景：**
- 检查奖励分布形状
- 识别偏斜或异常值
- 验证裁剪效果

### 9.4 仪表板可视化

结合所有可视化类型的综合视图：

```python
trace.plot_dashboard(save_path="dashboard.png")
```

包括：
- 奖励热图
- 相关性矩阵
- 总奖励趋势
- 各组件趋势

## 10. 课程学习设计模式

### 10.1 基于 Episode 的渐进

```python
Stage("easy", episodes=100)
Stage("medium", episodes=200)
Stage("hard")  # 无限制
```

**适用于**：固定长度训练，可预测的渐进

### 10.2 基于性能的渐进

```python
Stage("medium", condition=lambda: success_rate > 0.7)
Stage("hard", condition=lambda: avg_reward > 5.0)
```

**适用于**：自适应训练，质量导向的渐进

### 10.3 混合渐进

```python
Stage("intermediate",
    episodes=100,  # 最小值
    condition=lambda: success_rate > 0.5)  # 质量门槛
```

**适用于**：平衡方法，确保最小暴露

### 10.4 组件渐进

在各阶段逐渐添加奖励组件：

```python
# 阶段1：只有基础
stage1.add(1.0, name="reach_target")

# 阶段2：添加速度
stage2.add(1.0, name="reach_target")
stage2.add(0.5, name="speed_bonus")

# 阶段3：添加效率
stage3.add(1.0, name="reach_target")
stage3.add(0.5, name="speed_bonus")
stage3.add(0.3, name="efficiency")
```

**适用于**：逐步教授智能体多项技能
