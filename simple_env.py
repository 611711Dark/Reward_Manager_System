# simple_env.py
import numpy as np
from reward_system import RewardMgr   # 替换旧 RewardManager


class SimpleNavigationEnv:
    def __init__(self):
        self.max_x = 10.0
        self.max_y = 10.0
        self.max_speed = 5.0
        self.target_x = 8.0
        self.target_y = 8.0
        self.reset()

    def reset(self):
        self.x = np.random.uniform(0, 2)
        self.y = np.random.uniform(0, 2)
        self.speed = np.random.uniform(0, self.max_speed)
        self.direction_error = np.random.uniform(-30, 30)
        return self.get_state()

    def get_state(self):
        return np.array([self.x, self.y, self.speed, self.direction_error])

    def step(self, action, use_log_reward: bool = False):
        dx, dy = action
        self.x += dx * 0.1
        self.y += dy * 0.1
        self.speed = min(np.linalg.norm([dx, dy]), self.max_speed)
        self.direction_error += np.random.uniform(-5, 5)
        self.direction_error = np.clip(self.direction_error, -30, 30)

        mgr = self.calculate_reward()
        reward = mgr.total_log() if use_log_reward else mgr.total_raw()

        distance = np.linalg.norm([self.x - self.target_x, self.y - self.target_y])
        done = distance < 0.5 or self.x > self.max_x or self.y > self.max_y
        return self.get_state(), reward, done

    def calculate_reward(self) -> RewardMgr:
        mgr = RewardMgr(base=10)

        mgr.add_value(500.0, name="base")          # 固定奖励

        mgr.add_value(1000.0, var=self.speed,
                      max_var=self.max_speed, mul=1.5, name="speed")

        mgr.add_value(-300.0, var=abs(self.direction_error),
                      max_var=30, mul=2.0, name="direction")

        distance = np.linalg.norm([self.x - self.target_x, self.y - self.target_y])
        max_d = np.linalg.norm([self.max_x, self.max_y])
        closeness = 1.0 - (distance / max_d)
        mgr.add_value(1000.0, var=closeness ** 0.5,
                      max_var=1.0, mul=2.0, name="distance")
        return mgr

    def render_status(self):
        d = np.linalg.norm([self.x - self.target_x, self.y - self.target_y])
        print(f"Pos ({self.x:.2f}, {self.y:.2f}) | Spd {self.speed:.2f} | "
              f"DirErr {self.direction_error:5.1f}° | Dist {d:.2f}")
