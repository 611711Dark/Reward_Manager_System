# Reward Manager 系统设计文档
[Full Design Document](Document.md)

## 1. 核心设计思想

### 1.1 优先级分层奖励机制

在`Reward`类中实现的核心计算公式：
```python
self._raw = param * (base ** rank)
```
其中：
- `param`：奖励参数，控制奖励的幅度
- `rank`：优先级等级，决定奖励的层级重要性
- `base`：基数（默认10），放大层级差异的系数

这种设计使得高等级奖励（如rank=3）自然主导低等级奖励（如rank=1），无需手动调整权重。

### 1.2 动态变量关联系统

在`RewardMgr.add()`方法中实现的动态调节：
```python
if var is not None:
    param *= (var / max_var) * mul
```
其中：
- `var`：当前变量值
- `max_var`：变量最大值
- `mul`：乘数因子（默认1.5）

## 2. 架构设计原理

### 2.1 对数输出模式

在`Reward.log`属性中实现的对数压缩：
```python
sign = -1.0 if self._raw < 0 else 1.0
return sign * math.log(abs(self._raw) + 1, self.base)
```
这种设计：
- 保持原始值的符号
- 对绝对值取对数压缩
- 加1防止对0取对数

### 2.2 分层聚合机制

在`RewardTrace.to_reward_mgr()`中实现的多级聚合：
```python
for name in all_names:
    total = 0.0
    for rec in self._buf:
        total += rec["named"].get(name, 0.0)
    mgr.add_value(total / n_steps, name=name)
```
这种设计：
- 保留所有命名奖励组件
- 计算每个组件的平均值
- 创建新的RewardMgr实例

### 2.3 非线性距离奖励

在`simple_env.py`中实现的导航奖励：
```python
distance = np.linalg.norm([self.x - self.target_x, self.y - self.target_y])
max_d = np.linalg.norm([self.max_x, self.max_y])
closeness = 1.0 - (distance / max_d)
mgr.add_value(1000.0, var=closeness ** 0.5, max_var=1.0, mul=2.0, name="distance")
```
使用平方根变换实现非线性奖励增长。

## 3. 工程实现特色

### 3.1 内存优化

使用`__slots__`减少内存占用：
```python
class Reward:
    __slots__ = ("rank", "param", "base", "name", "_raw")
```

### 3.2 类型安全与链式API

类型注解和链式调用设计：
```python
def add(
    self,
    rank: int,
    param: float,
    var: Optional[float] = None,
    max_var: float = 1.0,
    mul: float = 1.0,
    name: Optional[str] = None,
) -> RewardMgr:  # 返回自身类型，支持链式调用
```

### 3.3 数值稳定性处理

处理极小值的保护机制：
```python
if abs(value) < 1e-9:
    rank, param = 0, 0.0
```

## 4. 核心组件详解

### 4.1 Reward类

```python
class Reward:
    def __init__(self, rank: int, param: float, base: int = 10, name: Optional[str] = None):
        self.rank = rank
        self.param = param
        self.base = base
        self.name = name
        self._raw = param * (base ** rank)
```

属性：
- `raw`: 原始奖励值
- `log`: 对数压缩后的奖励值

### 4.2 RewardMgr类

主要方法：
- `add()`: 手动添加奖励组件
- `add_value()`: 自动分解值到rank/param
- `total_raw()`: 计算原始奖励总和
- `total_log()`: 计算对数奖励总和

### 4.3 RewardTrace类

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
- `to_reward_mgr()`: 聚合轨迹为单个RewardMgr

## 5. 应用场景示例

### 5.1 导航环境集成

在`simple_env.py`中的实现：
```python
class SimpleNavigationEnv:
    def calculate_reward(self) -> RewardMgr:
        mgr = RewardMgr(base=10)
        mgr.add_value(500.0, name="base")
        mgr.add_value(1000.0, var=self.speed, max_var=self.max_speed, mul=1.5, name="speed")
        # ...其他奖励组件
        return mgr
```

### 5.2 多级监控系统

在`demo.py`中实现的三级监控：
```python
# Step级监控
step_arrays = final_step_trace.arrays()
axes[0].plot(step_arrays["raw"], label="raw")

# Game级监控
game_arrays = final_game_trace.arrays()
axes[1].plot(game_arrays["log"], label="log")

# Episode级监控
ep_arrays = episode_hist.arrays()
axes[2].plot(ep_arrays["distance"], label="distance")
```

## 6. 扩展与定制

### 6.1 自定义聚合策略

扩展`RewardTrace`类：
```python
class CustomRewardTrace(RewardTrace):
    def to_reward_mgr(self, mode='avg'):
        if mode == 'max':
            # 最大值聚合
        elif mode == 'min':
            # 最小值聚合
```

### 6.2 时间衰减机制

在添加奖励时应用衰减：
```python
decay_factor = math.exp(-0.01 * step_count)
mgr.add(rank, param * decay_factor, name="time_sensitive")
```

### 6.3 多智能体协同

团队奖励分配示例：
```python
team_reward = RewardMgr()
for agent in agents:
    individual_reward = agent.calculate_reward()
    team_reward.add_value(individual_reward.total_raw() * 0.7)
team_reward.add_value(global_bonus, name="team_bonus")
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
reward_trace = RewardTrace(maxlen=1000)  # 只保留最近的1000条记录
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
| `base` | int | 10 | 层级差异基数，越大差异越显著 |
| `rank` | int | - | 优先级等级，关键指标建议"5<=rank<=10" |
| `mul` | float | 1.5 | 动态变量乘数，调整奖励幅度 |
| `maxlen` | int | None | 历史记录最大长度 |

实际应用示例：
```python
# 安全关键型应用
mgr = RewardMgr(base=10)
mgr.add(rank=3, param=-2.0, name="collision_penalty")  # 高优先级惩罚

# 性能优化应用
mgr.add(rank=2, param=1.5, var=speed, max_var=max_speed, mul=1.2, name="speed_bonus")
```
