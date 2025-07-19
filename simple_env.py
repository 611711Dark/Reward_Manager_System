# simple_env.py
import random
import numpy as np
from reward_manager import RewardManager


class SimpleNavigationEnv:
    """Simple navigation environment example"""

    def __init__(self):
        """Initialize environment"""
        # Environment parameters
        self.max_x = 10.0
        self.max_y = 10.0
        self.max_speed = 5.0
        self.target_x = 8.0
        self.target_y = 8.0

        # State
        self.x = 0.0
        self.y = 0.0
        self.speed = 0.0
        self.direction_error = 0.0  # Direction deviation in degrees

        self.reset()

    def reset(self):
        """Reset environment"""
        self.x = random.uniform(0, 2)
        self.y = random.uniform(0, 2)
        self.speed = random.uniform(0, self.max_speed)
        self.direction_error = random.uniform(-30, 30)  # -30 to 30 degree deviation
        return self.get_state()

    def get_state(self):
        """Get current state"""
        return np.array([self.x, self.y, self.speed, self.direction_error])

    def step(self, action, use_log_reward: bool = False):
        """Execute action and return reward"""
        # Simple action processing (example only)
        dx, dy = action[0], action[1]

        # Update position
        self.x += dx * 0.1
        self.y += dy * 0.1

        # Update speed (based on action magnitude)
        self.speed = min(np.linalg.norm([dx, dy]), self.max_speed)

        # Update direction error (random change)
        self.direction_error += random.uniform(-5, 5)
        self.direction_error = np.clip(self.direction_error, -30, 30)

        # Calculate reward
        reward, manager = self.calculate_reward()
        reward = manager.total_value(use_log_reward)

        # Check termination
        distance_to_target = np.sqrt((self.x - self.target_x)**2 + (self.y - self.target_y)**2)
        done = distance_to_target < 0.5 or self.x > self.max_x or self.y > self.max_y

        return self.get_state(), reward, done

    def calculate_reward(self):
        """Calculate reward using RewardManager"""
        manager = RewardManager(base=10)

        # 1. Base position reward (fixed)
        manager.add_reward(2, 5.0)  # (2,5.0) = 500

        # 2. Speed reward (speed-dependent)
        manager.add_reward(
            rank=3,
            param=1.0,
            associated_var=self.speed,
            max_var_value=self.max_speed,
            multiplier=1.5
        )  # At max: 1000 * (5/5) * 1.5 = 1500

        # 3. Direction error penalty (error-dependent)
        manager.add_reward(
            rank=2,
            param=-3.0,
            associated_var=abs(self.direction_error),
            max_var_value=30,
            multiplier=2.0
        )  # At max error: -300 * (30/30) * 2.0 = -600

        # 4. Target distance reward (improved for clearer distance impact)
        distance = np.sqrt((self.x - self.target_x)**2 + (self.y - self.target_y)**2)
        max_distance = np.sqrt(self.max_x**2 + self.max_y**2)

        # Use nonlinear transform to make close-range rewards increase faster
        closeness = 1.0 - (distance / max_distance)  # Between 0 and 1
        nonlinear_closeness = closeness ** 0.5  # Square root transform

        manager.add_reward(
            rank=3,
            param=5.0,  # Increased base parameter
            associated_var=nonlinear_closeness,
            max_var_value=1.0,
            multiplier=2.0
        )

        return manager.total_value(False), manager

    def render_status(self):
        """Display current status"""
        distance = np.sqrt((self.x - self.target_x)**2 + (self.y - self.target_y)**2)
        print(f"Position: ({self.x:.2f}, {self.y:.2f})")
        print(f"Speed: {self.speed:.2f} m/s")
        print(f"Direction Error: {self.direction_error:.1f}°")
        print(f"Distance to Target: {distance:.2f}")
