# demo.py
import random
import numpy as np
from simple_env import SimpleNavigationEnv

def compare_log_modes():
    """Compare logarithmic vs normal reward modes"""
    print("=== Logarithmic Mode vs Normal Mode Comparison ===\n")

    # Create environment
    env = SimpleNavigationEnv()

    # Set identical state for comparison
    env.x = 3.0
    env.y = 4.0
    env.speed = 4.5
    env.direction_error = -15.0

    print("Environment Status:")
    env.render_status()
    print()

    # Calculate rewards
    normal_reward, normal_manager = env.calculate_reward()
    log_reward = normal_manager.total_value(True)

    print("Normal Mode:")
    print(f"Total Reward: {normal_reward:.1f}")
    print("Detailed Rewards:")
    for i, reward in enumerate(normal_manager.rewards):
        print(f"  {i+1}. {reward} (log={reward.log_value():.3f})")
    print()

    print("Logarithmic Mode:")
    print(f"Total Reward: {log_reward:.3f}")
    print()

    print(f"Value Range Difference: Normal {normal_reward:.0f} vs Log {log_reward:.3f}")
    print(f"Value Scaling Factor: {abs(normal_reward / log_reward):.0f}x")


def run_environment_demo():
    """Run environment demonstration"""
    print("\n=== Environment Interaction Demo ===\n")

    # Create environment
    env = SimpleNavigationEnv()
    state = env.reset()

    print(f"Target Position: ({env.target_x}, {env.target_y})")
    print("Initial Status:")
    env.render_status()
    print()

    # Run several steps
    for step in range(5):
        print(f"--- Step {step + 1} ---")

        # Simple strategy: move toward target
        dx = env.target_x - env.x
        dy = env.target_y - env.y
        norm = np.sqrt(dx*dx + dy*dy)
        if norm > 0:
            dx, dy = dx/norm, dy/norm

        # Add some randomness
        dx += random.uniform(-0.3, 0.3)
        dy += random.uniform(-0.3, 0.3)

        action = [dx, dy]
        state, reward, done = env.step(action, use_log_reward=True)

        print(f"Action: [{dx:.2f}, {dy:.2f}]")
        env.render_status()
        print(f"Reward: {reward:.3f}")

        if done:
            print("Environment terminated!")
            break
        print()


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    compare_log_modes()
    run_environment_demo()
