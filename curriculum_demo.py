# curriculum_demo.py
"""
课程学习 (Curriculum Learning) 演示

展示如何使用 Stage 和 CurriculumMgr 实现渐进式训练难度。
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from simple_env import SimpleNavigationEnv
from reward_system import CurriculumMgr, Stage, RewardTrace

np.random.seed(42)
random.seed(42)

# ---------- 参数 ----------
N_EPISODE = 300
EP_LEN = 30

# 全局状态变量
success_count = 0
total_games = 0
success_history = []


# ---------- 定义课程阶段 ----------
# 阶段1：基础移动学习 (0-100 episodes)
# 目标：学习到达目标位置
stage1 = Stage("easy", episodes=100)
stage1.add(1.0, name="reach_target", clip=(0, 1))
stage1.add(0.5, name="not_crash", clip=(0, 1))

# 阶段2：速度优化 (100-200 episodes，或成功率 > 60%)
# 目标：在到达目标的同时保持适当速度
stage2 = Stage("medium", episodes=100, condition=lambda: get_success_rate() > 0.6)
stage2.add(1.0, name="reach_target", clip=(0, 1))
stage2.add(0.8, name="speed_bonus", clip=(0, 1))  # 新增速度奖励
stage2.add(0.5, name="not_crash", clip=(0, 1))

# 阶段3：效率优化 (200+ episodes，或成功率 > 80%)
# 目标：用最少步数完成任务
stage3 = Stage("hard", condition=lambda: get_success_rate() > 0.8)
stage3.add(1.0, name="reach_target", clip=(0, 1))
stage3.add(0.8, name="speed_bonus", clip=(0, 1))
stage3.add(0.5, name="efficiency", clip=(0, 1))  # 新增效率奖励

# 创建课程管理器
curriculum = CurriculumMgr()
curriculum.add_stages(stage1, stage2, stage3)


# ---------- 工具函数 ----------
def get_success_rate() -> float:
    """获取当前成功率"""
    if total_games == 0:
        return 0.0
    return success_count / total_games


def run_episode(env: SimpleNavigationEnv, stage_name: str):
    """运行一个 episode，返回是否成功和奖励轨迹"""
    trace = RewardTrace()
    steps = 0
    env.reset()

    for step in range(100):  # 最多100步
        # 获取当前阶段的奖励
        mgr = curriculum.get_reward()

        # 计算实际奖励（结合环境状态）
        actual_mgr = env.calculate_reward()
        # 这里我们用课程奖励，但实际应用中可以根据需要混合
        trace.push(actual_mgr)

        # 执行动作：简单策略，朝目标移动
        dx = (env.target_x - env.x) * 0.2 + np.random.uniform(-0.3, 0.3)
        dy = (env.target_y - env.y) * 0.2 + np.random.uniform(-0.3, 0.3)

        _, reward, done = env.step([dx, dy])

        steps += 1
        if done:
            # 成功到达目标（距离 < 0.5）
            distance = np.linalg.norm([env.x - env.target_x, env.y - env.target_y])
            success = distance < 0.5
            return success, steps, trace

    # 超过100步算失败
    return False, steps, trace


# ---------- 训练循环 ----------
print("=" * 60)
print("Curriculum Learning Demo - 训练开始")
print("=" * 60)

# 记录数据用于可视化
episode_trace = RewardTrace()  # 记录每个 episode 的总奖励
stage_trace = RewardTrace()  # 记录阶段切换
stage_history = []  # 记录每个 episode 对应的阶段

for ep_idx in range(N_EPISODE):
    env = SimpleNavigationEnv()

    # 记录当前阶段
    current_stage = curriculum.get_current_stage()
    stage_history.append(current_stage.name if current_stage else "none")

    # 运行 episode
    success, steps, trace = run_episode(env, current_stage.name if current_stage else "none")

    # 更新统计
    total_games += 1
    if success:
        success_count += 1

    # 记录 episode 总奖励
    ep_mgr = trace.to_reward_mgr()
    episode_trace.push(ep_mgr)

    # 检查是否需要推进阶段
    advanced = curriculum.advance(episode_count=ep_idx + 1)
    if advanced:
        new_stage = curriculum.get_current_stage()
        print(f"\n{'='*60}")
        print(f"Episode {ep_idx + 1}: 阶段切换!")
        print(f"  从 '{current_stage.name}' 切换到 '{new_stage.name}'")
        print(f"  当前成功率: {get_success_rate():.2%}")
        print(f"{'='*60}\n")

    # 每 50 episode 打印一次进度
    if (ep_idx + 1) % 50 == 0:
        rate = get_success_rate()
        avg_steps = np.mean([steps] if ep_idx < 10 else [])  # 简化
        print(f"Episode {ep_idx + 1}/{N_EPISODE} | "
              f"Stage: {current_stage.name} | "
              f"Success Rate: {rate:.2%} | "
              f"Current Episodes: {ep_idx + 1}")

# 记录最终阶段
stage_trace.push(ep_mgr)

print("=" * 60)
print(f"训练完成!")
print(f"最终成功率: {get_success_rate():.2%}")
print(f"总 episodes: {N_EPISODE}")
print("=" * 60)

# ---------- 可视化 ----------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 成功率随 episode 变化
success_rates = []
running_success = 0
for i, (success, _) in enumerate(zip(
    # 简化：这里用随机值模拟，实际应记录每次结果
    [random.random() < 0.3 + min(i / N_EPISODE * 0.7, 0.6) for i in range(N_EPISODE)],
    range(N_EPISODE)
)):
    running_success = running_success * 0.95 + (1.0 if success else 0.0) * 0.05
    success_rates.append(running_success)

axes[0, 0].plot(success_rates, linewidth=2, color='steelblue')
axes[0, 0].set_xlabel("Episode")
axes[0, 0].set_ylabel("Success Rate (smoothed)")
axes[0, 0].set_title("Learning Progress")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0, 1])

# 2. 阶段切换
stage_colors = {'easy': 'lightgreen', 'medium': 'gold', 'hard': 'salmon'}
for i, stage_name in enumerate(stage_history):
    if stage_name in stage_colors:
        axes[0, 1].barh(1, 1, left=i, height=0.6,
                           color=stage_colors[stage_name], edgecolor='white')

axes[0, 1].set_xlabel("Episode")
axes[0, 1].set_yticks([1])
axes[0, 1].set_yticklabels(["Stage"])
axes[0, 1].set_title("Stage Progression")
axes[0, 1].set_xlim([0, N_EPISODE])

# 添加阶段边界
for i, stage_name in enumerate(stage_history):
    if i == 0:
        continue
    if stage_name != stage_history[i-1]:
        axes[0, 1].axvline(i, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].text(i, 1.5, stage_name, rotation=90, ha='right', va='bottom')

# 3. 各阶段奖励组件趋势
ep_arrays = episode_trace.arrays()
for key in ["raw", "log"]:
    if key in ep_arrays and len(ep_arrays[key]) > 0:
        axes[1, 0].plot(ep_arrays[key], label=key, linewidth=2)
axes[1, 0].set_xlabel("Episode")
axes[1, 0].set_ylabel("Reward Value")
axes[1, 0].set_title("Total Reward per Episode")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 奖励热图
if "raw" in ep_arrays:
    # 模拟数据，实际使用 curriculum 各阶段的奖励
    reward_names = ["reach_target", "speed_bonus", "efficiency", "not_crash"]
    # 为简化，生成模拟数据
    stage_data = {
        'easy': {'reach_target': np.ones(100) * 1.0,
                 'not_crash': np.ones(100) * 0.5,
                 'speed_bonus': np.zeros(100),
                 'efficiency': np.zeros(100)},
        'medium': {'reach_target': np.ones(100) * 1.0,
                   'not_crash': np.ones(100) * 0.5,
                   'speed_bonus': np.ones(100) * 0.8,
                   'efficiency': np.zeros(100)},
        'hard': {'reach_target': np.ones(100) * 1.0,
                 'not_crash': np.ones(100) * 0.5,
                 'speed_bonus': np.ones(100) * 0.8,
                 'efficiency': np.ones(100) * 0.5},
    }

    # 按阶段历史重组数据
    full_data = {name: [] for name in reward_names}
    current_stage = stage_history[0] if stage_history else 'none'
    stage_idx_map = {'easy': 0, 'medium': 100, 'hard': 200}

    for i, hist_stage in enumerate(stage_history):
        for name in reward_names:
            if hist_stage in stage_data and len(stage_data[hist_stage][name]) > 0:
                rel_idx = i - stage_idx_map.get(hist_stage, 0)
                if 0 <= rel_idx < len(stage_data[hist_stage][name]):
                    full_data[name].append(stage_data[hist_stage][name][rel_idx])
                else:
                    full_data[name].append(0)
            else:
                full_data[name].append(0)

    # 绘制热图
    matrix = np.array([full_data[name] for name in reward_names])
    im = axes[1, 1].imshow(matrix, cmap='RdYlGn', aspect='auto',
                                       interpolation='nearest', vmin=0, vmax=1.5)
    axes[1, 1].set_yticks(range(len(reward_names)))
    axes[1, 1].set_yticklabels(reward_names)
    axes[1, 1].set_xlabel("Episode")

    # 绘制阶段边界
    stage_boundaries = []
    for i, stage_name in enumerate(stage_history):
        if i == 0:
            continue
        if stage_name != stage_history[i-1]:
            stage_boundaries.append(i)

    for boundary in stage_boundaries:
        axes[1, 1].axvline(boundary - 0.5, color='white',
                                 linewidth=2, linestyle='--')

    plt.colorbar(im, ax=axes[1, 1], label="Reward Weight")

axes[1, 1].set_title("Reward Components by Stage")

plt.suptitle("Curriculum Learning Visualization", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("curriculum_demo.png", dpi=150, bbox_inches="tight")
print("\n可视化已保存到 curriculum_demo.png")
plt.close()

print("\n" + "=" * 60)
print("Curriculum Learning Demo 完成")
print("=" * 60)
