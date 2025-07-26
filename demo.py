# demo.py
import random
import numpy as np
import matplotlib.pyplot as plt
from simple_env import SimpleNavigationEnv
from reward_system import RewardTrace, RewardMgr

np.random.seed(42)
random.seed(42)

# ---------- 参数 ----------
N_EPISODE = 60
EP_LEN    = 50

# ---------- 工具 ----------
def run_episode(env):
    """返回 step_trace"""
    trace = RewardTrace()
    env.reset()
    while True:
        dx = np.random.uniform(-0.5, 0.5) + (env.target_x - env.x) * 0.1
        dy = np.random.uniform(-0.5, 0.5) + (env.target_y - env.y) * 0.1
        _, _, done = env.step([dx, dy], use_log_reward=True)
        trace.push(env.calculate_reward())
        if done:
            break
    return trace

# ---------- 训练 ----------
env = SimpleNavigationEnv()

# 三级历史（仅末局/末 episode 保留）
final_step_trace   = None   # 最后一局
final_game_trace   = None   # 最后一个 episode 的 50 局
episode_hist       = RewardTrace()   # 100 episode 最终历史

game_trace = RewardTrace()

for ep_idx in range(N_EPISODE):
    game_trace.clear()  # 每 episode 清空 game 级
    for game_idx in range(EP_LEN):
        step_trace = run_episode(env)
        game_mgr = step_trace.to_reward_mgr()

        # 是否最后一局？
        if ep_idx == N_EPISODE - 1 and game_idx == EP_LEN - 1:
            final_step_trace = step_trace   # 保留最后一局
        else:
            step_trace.clear()              # 非末局立即清空

        # 推入 game 级
        game_trace.push(game_mgr)

    # episode 压缩
    ep_mgr = game_trace.to_reward_mgr()

    # 是否最后一个 episode？
    if ep_idx == N_EPISODE - 1:
        final_game_trace = game_trace       # 保留最后一个 episode
    else:
        game_trace.clear()                  # 非末 episode 立即清空

    # 推入 episode 级
    episode_hist.push(ep_mgr)

# ---------- 绘图 ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

# (1) 最后一局 Step 奖励
step_arrays = final_step_trace.arrays()
axes[0].set_title("Step-Level Reward (Last Game)")
for key in ["raw", "log", "base", "speed", "direction", "distance"]:
    if key in step_arrays:
        axes[0].plot(step_arrays[key], label=key)
axes[0].set_xlabel("Step")
axes[0].legend()

# (2) 最后一个 episode 的 50 局奖励
game_arrays = final_game_trace.arrays()
axes[1].set_title("Game-Level Reward (Last Episode)")
for key in ["raw", "log", "base", "speed", "direction", "distance"]:
    if key in game_arrays:
        axes[1].plot(game_arrays[key], label=key)
axes[1].set_xlabel("Game")
axes[1].legend()

# (3) 训练全过程 60 episode 奖励
ep_arrays = episode_hist.arrays()
axes[2].set_title("Episode Reward (60 Episodes)")
for key in ["raw", "log", "base", "speed", "direction", "distance"]:
    if key in ep_arrays:
        axes[2].plot(ep_arrays[key], label=key)
axes[2].set_xlabel("Episode")
axes[2].legend()

plt.tight_layout()
plt.show()
