# Reward Priority Manager (奖励优先级管理器)

[English Version](README.md) | [完整设计文档](Document_cn.md)

## 项目概述

Reward Priority Manager (RPM) 是一个创新的奖励管理系统，专为强化学习和复杂决策系统设计。它通过**分层优先级架构**和**动态变量关联**机制，解决了传统奖励工程中权重调节困难、奖励信号不稳定等核心问题。

RPM 的核心思想是将奖励值分解为 `rank`（等级）和 `param`（参数）两个维度：
```
最终奖励值 = 参数 × (基数^等级)
```
这种设计使高优先级奖励自然主导低优先级奖励，无需人工微调权重系数。

## 核心特性

1. **分层优先级架构**  
   - 高等级奖励自动主导低等级奖励
   - 支持 `rank/param` 直接控制或 `value` 自动分解
   - 基数可配置（默认10）

2. **动态变量关联**  
   ```python
   # 速度奖励：根据当前速度动态调节
   mgr.add(3, 1.0, var=current_speed, max_var=max_speed, mul=1.5, name="speed")
   ```

3. **多级聚合压缩**  
   ```mermaid
   graph TD
     A[Step级奖励] -->|50步| B[Game级聚合]
     B -->|50局| C[Episode级聚合]
     C -->|60章节| D[训练分析]
   ```

4. **双模式输出**  
   - `raw`：原始奖励值（保持量级差异）
   - `log`：对数压缩值（适合神经网络训练）

## 安装与使用

### 安装
```bash
git clone https://github.com/611711Dark/Reward_Manager_System.git
```

### 基础用法
```python
from reward_system import RewardMgr

# 创建奖励管理器
mgr = RewardMgr(base=10)

# 添加固定基础奖励
mgr.add_value(500.0, name="base")

# 添加动态速度奖励（当前速度5.0，最大速度10.0）
mgr.add_value(1000.0, var=5.0, max_var=10.0, mul=1.5, name="speed")

print(f"原始奖励: {mgr.total_raw():.1f}")#原始奖励: 1250.0
print(f"对数奖励: {mgr.total_log():.3f}")#对数奖励: 5.575
print(f"速度组件: {mgr['speed']:.1f}")#速度组件: 750.0
```

### 环境集成
```python
from simple_env import SimpleNavigationEnv

env = SimpleNavigationEnv()
state = env.reset()

# 执行动作并获取奖励
action = [0.5, 0.3]
next_state, reward, done = env.step(action, use_log_reward=True)
```

## 核心组件

### 1. Reward (原子奖励)
```python
r = Reward(rank=2, param=1.5, base=10, name="critical")
print(r.raw)  # 1.5 * 10² = 150.0
print(r.log)  # 2.1789769472931693
```

### 2. RewardMgr (奖励管理器)
```python
mgr = RewardMgr()
mgr.add_value(200.0, name="bonus")  # 自动分解rank/param
mgr.add(rank=1, param=3.0, name="penalty")  # 手动指定(推荐)

# 链式调用
mgr.add_value(500.0, name="base").add_value(-100.0, name="error")
```

### 3. RewardTrace (奖励轨迹)
```python
trace = RewardTrace()

# 记录多步奖励
for _ in range(10):
    mgr = env.calculate_reward()
    trace.push(mgr)

# 压缩为单一RewardMgr
summary = trace.to_reward_mgr()
```

## 三级监控系统

### 执行演示
```bash
python demo.py
```

### 可视化输出
![奖励监控系统](reward_system_demo.png)

1. **步骤级监控**  
   - 最后一局游戏的详细奖励组件
   - 包括原始值和对数压缩值

2. **游戏级监控**  
   - 最后一章节的50局游戏聚合
   - 显示各奖励组件的变化趋势

3. **章节级监控**  
   - 整个训练过程的60章节趋势
   - 识别长期奖励变化模式

## 设计优势

1. **数学可解释性**  
   奖励值 = 参数 × (基数^等级) 提供清晰的数学基础

2. **动态优先级**  
   ```python
   # 自动计算合适等级
   rank = max(0, int(math.log10(abs(value)/base)) + 1)
   ```

3. **内存高效**  
   - `__slots__` 减少内存占用
   - `deque` 实现滑动窗口

4. **多级分析**  
   ```python
   # 三级数据保留策略
   if ep_idx == N_EPISODE - 1:
       final_game_trace = game_trace  # 保留末章节
   ```

## 应用场景

1. **强化学习系统**  
   - 替代传统标量奖励
   - 解决奖励稀疏问题

2. **游戏AI开发**  
   - 复杂行为奖励组合
   - 多目标平衡

3. **机器人控制**  
   - 安全约束优先级
   - 多传感器奖励融合

## 贡献指南

欢迎通过 issue 或 pull request 贡献：
1. 报告问题或建议
2. 添加新环境示例
3. 扩展可视化功能
4. 优化核心算法

## 许可证

本项目采用 [MIT 许可证](LICENSE)。
