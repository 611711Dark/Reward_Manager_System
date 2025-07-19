# Reward Manager 系统 - 中文文档

## 目录
1. [系统概述](#系统概述)  
2. [核心特性](#核心特性)  
3. [安装指南](#安装指南)  
4. [基础使用](#基础使用)  
5. [API参考](#api参考)  
6. [集成示例](#集成示例)  
7. [高级用法](#高级用法)  
8. [许可证](#许可证)  
9. [参与贡献](#参与贡献)  
10. [英文文档](#英文文档)

## 系统概述

Reward Manager 是一个基于优先级的奖励管理系统，专为强化学习环境设计。它通过分层奖励架构和动态变量关联机制，提供灵活且直观的奖励计算方案。

## 核心特性

- **优先级奖励体系**：高优先级奖励自动主导低优先级奖励
- **动态变量关联**：奖励值与环境变量动态关联
- **对数输出模式**：可选的对数转换输出，有效压缩大数值范围
- **透明奖励分析**：清晰的奖励组成分解
- **易集成**：轻松对接各类强化学习环境

## 安装指南

```bash
git clone https://github.com/yourusername/reward-manager.git
cd reward-manager
```

## 基础使用

### 创建奖励

```python
from reward_manager import RewardManager

# 创建管理器（默认基数10）
manager = RewardManager()

# 添加奖励（优先级等级，参数）
manager.add_reward(rank=2, param=3.0)  # 3.0 * 10^2 = 300
manager.add_reward(rank=1, param=-2.0) # -2.0 * 10^1 = -20

# 通过数值直接添加（自动分解）
manager.add_value(4500)  # 自动存储为 rank=3, param=4.5
```

### 变量关联

```python
speed = 3.0
max_speed = 5.0

# 添加速度相关奖励（基于当前速度缩放）
manager.add_reward(
    rank=2,
    param=1.0,
    associated_var=speed,
    max_var_value=max_speed,
    multiplier=2.0
)
```

### 奖励计算

```python
# 获取原始总值
raw_total = manager.total_value(use_log=False)

# 获取对数转换值
log_total = manager.total_value(use_log=True)

print(f"原始总值: {raw_total:.1f}")
print(f"对数总值: {log_total:.3f}")
```

## API参考

### `PriorityReward` 类

#### `__init__(self, rank: int, param: float, base: int = 10)`
- `rank`: 优先级等级（越高越重要）
- `param`: 奖励参数（可负值）
- `base`: 计算基数（默认10）

#### 方法:
- `raw_value() -> float`: 返回原始奖励值 (param * base^rank)
- `log_value() -> float`: 返回对数转换值 (sign * log10(abs(raw) + 1))

### `RewardManager` 类

#### `__init__(self, base: int = 10)`
- `base`: 奖励计算的数值基数（默认10）

#### 主要方法:

**添加奖励:**
- `add_reward(rank, param, associated_var=None, max_var_value=1.0, multiplier=1.0)`
  - 添加带可选变量缩放的奖励
- `add_value(value, associated_var=None, max_var_value=1.0, multiplier=1.0)`
  - 通过数值添加奖励（自动分解为rank/param）

**计算奖励:**
- `total_value(use_log=False) -> float`
  - 计算所有奖励的总和（可选对数转换）

**检查:**
- `highest_priority() -> Optional[PriorityReward]`
- `lowest_priority() -> Optional[PriorityReward]`
- `clear()` - 清除所有奖励

## 集成示例

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

        # 距离奖励（越近越好）
        distance = np.linalg.norm(self.position - self.target)
        max_dist = np.linalg.norm([10.0, 10.0])  # 最大可能距离
        closeness = 1.0 - (distance / max_dist)

        manager.add_reward(
            rank=3,
            param=5.0,
            associated_var=closeness,
            max_var_value=1.0,
            multiplier=2.0
        )

        # 速度奖励
        manager.add_reward(
            rank=2,
            param=1.5,
            associated_var=self.speed,
            max_var_value=self.max_speed
        )

        return manager.total_value(use_log=True), manager
```

## 高级用法

### 自定义基数

```python
# 使用基数2创建二进制风格奖励
binary_manager = RewardManager(base=2)
binary_manager.add_reward(5, 1.0)  # 1.0 * 2^5 = 32
```

### 奖励分析

```python
# 分析奖励组成
total, manager = env.calculate_reward()
print(f"总奖励: {total:.3f}")

print("详细分解:")
for i, reward in enumerate(manager):
    print(f"{i+1}. 等级 {reward.rank}: {reward.param:.2f} = {reward.raw_value():.1f} (对数: {reward.log_value():.3f})")
```

## 许可证

MIT 许可证

## 参与贡献

欢迎提交issue或pull request参与贡献！

## 英文文档

[点击查看完整英文文档](README.md)
