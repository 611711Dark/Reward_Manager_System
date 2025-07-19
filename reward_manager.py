# reward_manager.py
import math
from typing import Optional, List, Tuple


class PriorityReward:
    """Priority-based reward object"""

    def __init__(self, rank: int, param: float, base: int = 10):
        """
        Create priority reward
        :param rank: Priority level (integer)
        :param param: Reward parameter (float, supports negative values)
        :param base: Numerical base (default 10)
        """
        self.rank = rank
        self.param = param
        self.base = base

    def raw_value(self) -> float:
        """Calculate raw reward value: param * (base ** rank)"""
        return self.param * (self.base ** self.rank)

    def log_value(self) -> float:
        """
        Calculate logarithmic reward value
        Logarithmic form: sign(raw_value) * log10(abs(raw_value) + 1)
        """
        raw = self.raw_value()

        if abs(raw) < 1e-9:
            return 0.0

        # Preserve sign, take log of absolute value
        sign = 1 if raw >= 0 else -1
        log_value = math.log10(abs(raw) + 1)
        return sign * log_value

    def __lt__(self, other) -> bool:
        """Comparison operator: rank first, param second"""
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.param < other.param

    def __eq__(self, other) -> bool:
        return self.rank == other.rank and abs(self.param - other.param) < 1e-9

    def __repr__(self) -> str:
        raw = self.raw_value()
        return f"({self.rank},{self.param:.3f})={raw:.1f}"


class RewardManager:
    """Priority-based reward manager"""

    def __init__(self, base: int = 10):
        """
        Initialize manager
        :param base: Numerical base (default 10)
        """
        self.base = base
        self.rewards: List[PriorityReward] = []

    def add_reward(self,
                   rank: int,
                   param: float,
                   associated_var: Optional[float] = None,
                   max_var_value: float = 1.0,
                   multiplier: float = 1.0) -> 'RewardManager':
        """
        Add reward (supports variable scaling)
        :param rank: Priority level
        :param param: Reward parameter
        :param associated_var: Associated variable (optional)
        :param max_var_value: Maximum variable value
        :param multiplier: Scaling factor
        :return: Returns self for method chaining
        """
        if associated_var is None:
            # Base reward
            reward = PriorityReward(rank, param, self.base)
        else:
            # Scaled reward: final_param = param * (var/max_var) * multiplier
            scale_factor = (associated_var / max_var_value) * multiplier
            final_param = param * scale_factor
            reward = PriorityReward(rank, final_param, self.base)

        self.rewards.append(reward)
        return self

    def add_value(self,
                  value: float,
                  associated_var: Optional[float] = None,
                  max_var_value: float = 1.0,
                  multiplier: float = 1.0) -> 'RewardManager':
        """
        Add reward by value (automatically decomposed to rank and param)
        """
        # Decompose value into (rank, param)
        if abs(value) < 1e-9:
            rank, param = 0, 0.0
        else:
            abs_value = abs(value)
            rank = max(0, int(math.log10(abs_value / self.base) + 1))
            param = value / (self.base ** rank)

        return self.add_reward(rank, param, associated_var, max_var_value, multiplier)

    def total_value(self, use_log: bool = False) -> float:
        """Calculate total reward value"""
        if use_log:
            return sum(r.log_value() for r in self.rewards)
        return sum(r.raw_value() for r in self.rewards)

    def highest_priority(self) -> Optional[PriorityReward]:
        """Get highest priority reward"""
        return max(self.rewards) if self.rewards else None

    def lowest_priority(self) -> Optional[PriorityReward]:
        """Get lowest priority reward"""
        return min(self.rewards) if self.rewards else None

    def clear(self) -> 'RewardManager':
        """Clear all rewards"""
        self.rewards.clear()
        return self

    def __len__(self) -> int:
        return len(self.rewards)

    def __getitem__(self, index: int) -> PriorityReward:
        return self.rewards[index]

    def __repr__(self) -> str:
        rewards_str = ", ".join(str(r) for r in self.rewards)
        total = self.total_value(False)
        log_total = self.total_value(True)
        return f"RewardManager[{rewards_str}] Total={total:.1f} (LogTotal={log_total:.3f})"
