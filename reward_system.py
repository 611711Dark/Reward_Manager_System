# reward_system.py
from __future__ import annotations
import math
from collections import deque
from typing import List, Dict, Optional

# ---------- 单条奖励 ----------
class Reward:
    __slots__ = ("rank", "param", "base", "name", "_raw")

    def __init__(self, rank: int, param: float, base: int = 10, name: Optional[str] = None):
        self.rank = rank
        self.param = param
        self.base = base
        self.name = name
        self._raw = param * (base ** rank)

    @property
    def raw(self) -> float:
        return self._raw

    @property
    def log(self) -> float:
        sign = -1.0 if self._raw < 0 else 1.0
        return sign * math.log(abs(self._raw) + 1, self.base)

    def __repr__(self) -> str:
        name_part = f"'{self.name}'" if self.name else ""
        return f"({self.rank},{self.param:.3f}){name_part}={self._raw:.1f}"


# ---------- 单步奖励管理 ----------
class RewardMgr:
    def __init__(self, base: int = 10):
        self.base = base
        self._items: List[Reward] = []
        self._names: Dict[str, Reward] = {}

    def add(
        self,
        rank: int,
        param: float,
        var: Optional[float] = None,
        max_var: float = 1.0,
        mul: float = 1.5,
        name: Optional[str] = None,
    ) -> RewardMgr:
        if var is not None:
            param *= (var / max_var) * mul
        r = Reward(rank, param, self.base, name)
        if name is not None:
            if name in self._names:
                raise ValueError(f"Reward name '{name}' already exists.")
            self._names[name] = r
        self._items.append(r)
        return self

    def add_value(
        self,
        value: float,
        var: Optional[float] = None,
        max_var: float = 1.0,
        mul: float = 1.5,
        name: Optional[str] = None,
    ) -> RewardMgr:
        if abs(value) < 1e-9:
            rank, param = 0, 0.0
        else:
            rank = max(0, int(math.log10(abs(value) / self.base)) + 1)
            param = value / (self.base ** rank)
        return self.add(rank, param, var, max_var, mul, name)

    def total_raw(self) -> float:
        return sum(r.raw for r in self._items)

    def total_log(self) -> float:
        return sum(r.log for r in self._items)

    def __getitem__(self, name: str) -> float:
        return self._names[name].raw

    def clear(self) -> RewardMgr:
        self._items.clear()
        self._names.clear()
        return self

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        items = ", ".join(map(str, self._items))
        return f"<RewardMgr {items} raw={self.total_raw():.1f} log={self.total_log():.3f}>"


# ---------- 训练历史 ----------
class RewardTrace:
    def __init__(self, maxlen: Optional[int] = None):
        self._buf = deque(maxlen=maxlen)

    def push(self, mgr: RewardMgr) -> RewardTrace:
        self._buf.append(
            {
                "raw": mgr.total_raw(),
                "log": mgr.total_log(),
                "named": {k: v.raw for k, v in mgr._names.items()},
            }
        )
        return self

    def arrays(self) -> Dict[str, list]:
        if not self._buf:
            return {}
        keys = self._buf[-1]["named"].keys()
        return {
            "raw": [r["raw"] for r in self._buf],
            "log": [r["log"] for r in self._buf],
            **{k: [r["named"].get(k, math.nan) for r in self._buf] for k in keys},
        }

    def clear(self) -> RewardTrace:
        self._buf.clear()
        return self

    def __len__(self) -> int:
        return len(self._buf)

    # 新增：把轨迹压缩成 RewardMgr
    def to_reward_mgr(self, base: int = 10) -> RewardMgr:
        mgr = RewardMgr(base=base)
        if not self._buf:
            return mgr
        all_names = set()
        for rec in self._buf:
            all_names.update(rec["named"].keys())
        n_steps = len(self._buf)
        for name in all_names:
            total = 0.0
            for rec in self._buf:
                total += rec["named"].get(name, 0.0)
            mgr.add_value(total / n_steps, name=name)
        return mgr

    # 链式压缩
    def compress_into(self, target: "RewardTrace", base: int = 10) -> "RewardTrace":
        target.push(self.to_reward_mgr(base))
        return self
