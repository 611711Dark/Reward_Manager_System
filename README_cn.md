# Reward Manager System (奖励管理系统)

[English Version](README.md) | [完整设计文档](Document_cn.md)

## 项目概述

Reward Manager System (RMS) 是一个简洁的奖励管理系统，专为强化学习和复杂决策系统设计。它提供**直接的奖励值控制**、**动态变量关联**、**奖励裁剪**、**丰富的可视化**和**课程学习**支持。

核心设计简单直观：**直接输入奖励值**（如 0.1, 0.04, 5.0 等），可选地根据动态变量进行缩放。

## 核心特性

1. **直接值控制**
   - 直接输入奖励值（如 0.1, 0.04, 5.0）
   - 无需复杂的 rank/param 配置
   - 直观易懂，方便调参

2. **动态变量关联**
   ```python
   # 速度奖励：根据当前速度动态调节
   mgr.add(1.0, var=current_speed, max_var=max_speed, mul=1.5, name="speed")
   ```

3. **奖励裁剪**
   - 防止极端奖励值
   - 支持范围裁剪和单值裁剪

   ```python
   # 范围裁剪 [0, 5]
   mgr.add(10.0, name="bonus", clip=(0, 5))

   # 只限制上限
   mgr.add(10.0, name="reward", clip=5)
   ```

4. **课程学习**
   - 多阶段奖励递进
   - 支持基于 episode、game 和性能指标的触发

   ```python
   stage1 = Stage("easy", episodes=100)
   stage2 = Stage("medium", condition=lambda: success_rate > 0.8)
   ```

5. **丰富的可视化**
   - 奖励组件随时间的热图
   - 组件间相关性矩阵
   - 各组件分布直方图
   - 综合仪表板

   ```python
   trace.plot_heatmap(save_path="heatmap.png")
   trace.plot_correlation(save_path="correlation.png")
   trace.plot_distribution(save_path="distribution.png")
   trace.plot_dashboard(save_path="dashboard.png")
   ```

6. **多级聚合压缩**
   ```mermaid
   graph TD
     A[Step级奖励] -->|50步| B[Game级聚合]
     B -->|50局| C[Episode级聚合]
     C -->|60章节| D[训练分析]
   ```

7. **双模式输出**
   - `raw`：原始奖励值（保持量级差异）
   - `log`：对数压缩值（适合神经网络训练）

## 安装与使用

### 安装
```bash
git clone https://github.com/611711Dark/Reward_Manager_System.git
pip install numpy matplotlib  # 可视化功能需要
```

### 基础用法
```python
from reward_system import RewardMgr

# 创建奖励管理器
mgr = RewardMgr()

# 添加固定基础奖励
mgr.add(5.0, name="base")

# 添加带裁剪的动态速度奖励
mgr.add(3.0, var=5.0, max_var=10.0, mul=1.5, name="speed", clip=(0, 5))

print(f"原始奖励: {mgr.total_raw():.3f}")  # 原始奖励: 5.000
print(f"对数奖励: {mgr.total_log():.3f}")  # 对数奖励: 0.699
print(f"速度组件: {mgr['speed']:.3f}")    # 速度组件: 2.250
```

### 课程学习示例
```python
from reward_system import CurriculumMgr, Stage

# 定义训练阶段
curriculum = CurriculumMgr()

# 阶段1：基础移动（episodes 0-100）
stage1 = Stage("easy", episodes=100)
stage1.add(1.0, name="reach_target", clip=(0, 1))
stage1.add(0.5, name="not_crash", clip=(0, 1))

# 阶段2：速度控制（成功率 > 0.8 时触发）
stage2 = Stage("medium", condition=lambda: success_rate > 0.8)
stage2.add(1.0, name="reach_target", clip=(0, 1))
stage2.add(0.8, name="speed_bonus", var=speed/max_speed)
stage2.add(0.5, name="not_crash")

# 阶段3：完整任务（最终阶段）
stage3 = Stage("hard")
stage3.add(1.0, name="reach_target")
stage3.add(0.8, name="speed_bonus")
stage3.add(0.5, name="efficiency")

curriculum.add_stages(stage1, stage2, stage3)

# 训练循环
for ep in range(500):
    # 获取当前阶段奖励
    mgr = curriculum.get_reward()

    # 执行动作...
    state, reward, done = env.step(action)

    # 更新进度并检查是否切换阶段
    if curriculum.advance(episode_count=ep):
        print(f"切换到: {curriculum.get_current_stage().name}")
```

### 可视化示例
```python
from reward_system import RewardTrace

# 训练过程中记录奖励
trace = RewardTrace()
for step in range(100):
    mgr = env.calculate_reward()
    trace.push(mgr)

# 可视化
trace.plot_dashboard(save_path="dashboard.png")
trace.plot_heatmap(save_path="heatmap.png")
trace.plot_correlation(save_path="correlation.png")
trace.plot_distribution(save_path="distribution.png")
```

## 核心组件

### 1. Reward (原子奖励)
```python
r = Reward(0.5, name="critical")
print(r.raw)  # 0.5
print(r.log)  # 0.17609125905568124
```

### 2. RewardMgr (奖励管理器)
```python
mgr = RewardMgr()
mgr.add(2.0, name="bonus", clip=(0, 5))  # 带裁剪
mgr.add(-1.0, name="penalty", clip=(-10))  # 只限制上限

# 链式调用
mgr.add(5.0, name="base").add(-0.5, name="error")
```

### 3. Stage (训练阶段)
```python
# 基于 episode 触发
stage = Stage("easy", episodes=100)

# 基于性能指标触发
stage = Stage("medium", condition=lambda: success_rate > 0.8)

# 无触发条件（最终阶段）
stage = Stage("hard")

# 添加奖励到阶段
stage.add(1.0, name="reward", clip=(0, 2))
```

### 4. CurriculumMgr (课程管理器)
```python
curriculum = CurriculumMgr()
curriculum.add_stage(stage1).add_stage(stage2).add_stage(stage3)

# 或批量添加
curriculum.add_stages(stage1, stage2, stage3)

# 获取当前奖励
mgr = curriculum.get_reward()

# 检查并推进阶段
if curriculum.advance(episode_count=150):
    print("阶段已推进！")
```

### 5. RewardTrace (奖励轨迹)
```python
trace = RewardTrace()

# 记录多步奖励
for _ in range(10):
    mgr = env.calculate_reward()
    trace.push(mgr)

# 压缩为单一RewardMgr
summary = trace.to_reward_mgr()

# 可视化
trace.plot_heatmap()
trace.plot_correlation()
trace.plot_distribution()
trace.plot_dashboard()
```

## API 参考

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

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `value` | float | - | 直接奖励值（如 0.1, 0.04） |
| `var` | float | None | 动态变量值（可选） |
| `max_var` | float | 1.0 | 变量最大值，用于归一化 |
| `mul` | float | 1.0 | 倍数因子 |
| `name` | str | None | 奖励名称，用于查询 |
| `clip` | float/元组 | None | 裁剪限制：`max` 或 `(min, max)` |

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

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `name` | str | - | 阶段名称 |
| `episodes` | int | None | 达到此 episode 数量后触发 |
| `games` | int | None | 达到此 game 数量后触发 |
| `condition` | callable | None | 自定义条件函数 |

### CurriculumMgr.advance()
```python
def advance(
    self,
    episode_count: Optional[int] = None,
    game_count: Optional[int] = None,
    check_condition: bool = True,
) -> bool
```

如果阶段被推进返回 `True`，否则返回 `False`。

### RewardTrace 可视化方法

| 方法 | 描述 |
|------|------|
| `plot_heatmap(save_path=None, title="...")` | 奖励随时间的热图 |
| `plot_correlation(save_path=None, title="...")` | 组件相关性矩阵 |
| `plot_distribution(save_path=None, title="...")` | 分布直方图 |
| `plot_dashboard(save_path=None)` | 综合仪表板 |

## 应用场景

1. **强化学习系统**
   - 替代传统标量奖励
   - 解决奖励稀疏问题
   - 使用课程学习实现渐进难度

2. **游戏AI开发**
   - 复杂行为奖励组合
   - 多目标平衡
   - 逐步解锁高级机制

3. **机器人控制**
   - 安全约束优先级
   - 多传感器奖励融合
   - 从基础任务开始，过渡到复杂任务

## 演示文件

| 文件 | 描述 |
|------|------|
| `demo.py` | 三级监控演示 |
| `curriculum_demo.py` | 课程学习演示 |
| `simple_env.py` | 简单导航环境 |

运行演示：

```bash
python demo.py
python curriculum_demo.py
```

## 贡献指南

欢迎通过 issue 或 pull request 贡献：
1. 报告问题或建议
2. 添加新环境示例
3. 扩展可视化功能
4. 优化核心算法

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

---
