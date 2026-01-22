# reward_system.py
from __future__ import annotations
import math
from collections import deque
from typing import List, Dict, Optional, Callable, Union


# ---------- 单条奖励 ----------
class Reward:
    __slots__ = ("value", "name")

    def __init__(self, value: float, name: Optional[str] = None):
        self.value = value
        self.name = name

    @property
    def raw(self) -> float:
        return self.value

    @property
    def log(self) -> float:
        if abs(self.value) < 1e-9:
            return 0.0
        sign = -1.0 if self.value < 0 else 1.0
        return sign * math.log(abs(self.value) + 1, 10)

    def __repr__(self) -> str:
        name_part = f"'{self.name}'" if self.name else ""
        return f"{name_part}={self.value:.3f}"


# ---------- 单步奖励管理 ----------
class RewardMgr:
    def __init__(self):
        self._items: List[Reward] = []
        self._names: Dict[str, Reward] = {}

    def add(
        self,
        value: float,
        var: Optional[float] = None,
        max_var: float = 1.0,
        mul: float = 1.0,
        name: Optional[str] = None,
        clip: Optional[Union[float, tuple[float, float]]] = None,
    ) -> RewardMgr:
        """添加一个奖励

        Args:
            value: 奖励值（可直接输入如 0.1, 0.04 等）
            var: 动态变量值（如当前速度、距离等）
            max_var: 变量的最大值（用于归一化）
            mul: 倍数因子（默认 1.0 不放大）
            name: 奖励名称（用于查询）
            clip: 裁剪限制，可以是单个值（上限）或元组 (min, max)
        """
        if var is not None:
            value = value * (var / max_var) * mul

        # 裁剪处理
        if clip is not None:
            if isinstance(clip, tuple):
                min_val, max_val = clip
                if min_val is not None:
                    value = max(value, min_val)
                if max_val is not None:
                    value = min(value, max_val)
            else:  # 单个值表示上限
                value = min(value, clip)

        r = Reward(value, name)
        if name is not None:
            if name in self._names:
                raise ValueError(f"Reward name '{name}' already exists.")
            self._names[name] = r
        self._items.append(r)
        return self

    def total_raw(self) -> float:
        return sum(r.value for r in self._items)

    def total_log(self) -> float:
        return sum(r.log for r in self._items)

    def __getitem__(self, name: str) -> float:
        return self._names[name].value

    def clear(self) -> RewardMgr:
        self._items.clear()
        self._names.clear()
        return self

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        items = ", ".join(map(str, self._items))
        return f"<RewardMgr {items} raw={self.total_raw():.3f} log={self.total_log():.3f}>"

    # 兼容旧 API
    def add_value(
        self,
        value: float,
        var: Optional[float] = None,
        max_var: float = 1.0,
        mul: float = 1.0,
        name: Optional[str] = None,
    ) -> RewardMgr:
        """兼容旧 API，直接调用 add 方法"""
        return self.add(value, var, max_var, mul, name)


# ---------- 训练历史 ----------
class RewardTrace:
    def __init__(self, maxlen: Optional[int] = None):
        self._buf = deque(maxlen=maxlen)

    def push(self, mgr: RewardMgr) -> RewardTrace:
        self._buf.append(
            {
                "raw": mgr.total_raw(),
                "log": mgr.total_log(),
                "named": {k: v.value for k, v in mgr._names.items()},
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

    def to_reward_mgr(self) -> RewardMgr:
        mgr = RewardMgr()
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
            mgr.add(total / n_steps, name=name)
        return mgr

    def compress_into(self, target: "RewardTrace") -> "RewardTrace":
        target.push(self.to_reward_mgr())
        return self

    # ---------- 可视化功能 ----------
    def plot_heatmap(
        self,
        save_path: Optional[str] = None,
        title: str = "Reward Heatmap",
        cmap: str = "RdYlGn",
    ):
        """绘制奖励热图

        Args:
            save_path: 保存路径，None 则显示
            title: 图表标题
            cmap: 颜色映射
        """
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
        except ImportError as e:
            raise ImportError("可视化需要安装 numpy 和 matplotlib: pip install numpy matplotlib") from e

        if not self._buf:
            print("No data to plot.")
            return

        data = self.arrays()
        reward_names = [k for k in data.keys() if k not in ["raw", "log"]]

        if not reward_names:
            print("No named rewards to plot.")
            return

        # 构建矩阵
        matrix = np.array([[data[name][i] for name in reward_names] for i in range(len(data["raw"]))])
        matrix = matrix.T

        fig, ax = plt.subplots(figsize=(max(10, len(reward_names)), len(data["raw"]) / 5 + 2))

        im = ax.imshow(matrix, cmap=cmap, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(reward_names)))
        ax.set_yticklabels(reward_names)
        ax.set_xlabel("Step")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Reward Value")

        plt.title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Heatmap saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_correlation(
        self, save_path: Optional[str] = None, title: str = "Reward Correlation"
    ):
        """绘制奖励相关性矩阵

        Args:
            save_path: 保存路径，None 则显示
            title: 图表标题
        """
        try:
            import numpy as np
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("可视化需要安装 numpy 和 matplotlib") from e

        if not self._buf:
            print("No data to plot.")
            return

        data = self.arrays()
        reward_names = [k for k in data.keys() if k not in ["raw", "log"]]

        if len(reward_names) < 2:
            print("Need at least 2 named rewards for correlation.")
            return

        # 计算相关系数
        matrix = np.array([data[name] for name in reward_names])
        corr = np.corrcoef(matrix)

        fig, ax = plt.subplots(figsize=(max(6, len(reward_names)), len(reward_names)))

        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(len(reward_names)))
        ax.set_yticks(range(len(reward_names)))
        ax.set_xticklabels(reward_names, rotation=45, ha="right")
        ax.set_yticklabels(reward_names)

        # 添加数值标签
        for i in range(len(reward_names)):
            for j in range(len(reward_names)):
                text = ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correlation")

        plt.title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Correlation saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_distribution(
        self, save_path: Optional[str] = None, title: str = "Reward Distribution"
    ):
        """绘制奖励分布直方图

        Args:
            save_path: 保存路径，None 则显示
            title: 图表标题
        """
        try:
            import numpy as np
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("可视化需要安装 numpy 和 matplotlib") from e

        if not self._buf:
            print("No data to plot.")
            return

        data = self.arrays()
        reward_names = [k for k in data.keys() if k not in ["raw", "log"]]

        if not reward_names:
            print("No named rewards to plot.")
            return

        n_rewards = len(reward_names)
        n_cols = min(3, n_rewards)
        n_rows = (n_rewards + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
        if n_rewards == 1:
            axes = np.array([[axes]])
        axes = axes.flatten()

        for i, name in enumerate(reward_names):
            values = data[name]
            valid_values = [v for v in values if not math.isnan(v)]

            if valid_values:
                axes[i].hist(valid_values, bins=30, edgecolor="black", alpha=0.7)
                axes[i].axvline(np.mean(valid_values), color="red", linestyle="--", label=f"Mean: {np.mean(valid_values):.3f}")
                axes[i].set_xlabel("Value")
                axes[i].set_ylabel("Frequency")
                axes[i].set_title(name)
                axes[i].legend()
            else:
                axes[i].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[i].transAxes)
                axes[i].set_title(name)

        # 隐藏多余的子图
        for i in range(n_rewards, len(axes)):
            axes[i].axis("off")

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Distribution saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_dashboard(self, save_path: Optional[str] = None):
        """绘制综合仪表板（包含热图、相关性、分布）

        Args:
            save_path: 保存路径，None 则显示
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("可视化需要安装 matplotlib") from e

        if not self._buf:
            print("No data to plot.")
            return

        data = self.arrays()
        reward_names = [k for k in data.keys() if k not in ["raw", "log"]]

        if not reward_names:
            print("No named rewards to plot.")
            return

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5])

        # 1. 热图
        ax1 = fig.add_subplot(gs[0, 0])
        try:
            import numpy as np
            matrix = np.array([[data[name][i] for name in reward_names] for i in range(len(data["raw"]))])
            matrix = matrix.T
            im = ax1.imshow(matrix, cmap="RdYlGn", aspect="auto", interpolation="nearest")
            ax1.set_yticks(range(len(reward_names)))
            ax1.set_yticklabels(reward_names)
            ax1.set_xlabel("Step")
            ax1.set_title("Reward Heatmap")
            plt.colorbar(im, ax=ax1, label="Value")
        except Exception as e:
            ax1.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")

        # 2. 相关性
        ax2 = fig.add_subplot(gs[0, 1])
        if len(reward_names) >= 2:
            try:
                import numpy as np
                matrix = np.array([data[name] for name in reward_names])
                corr = np.corrcoef(matrix)
                im = ax2.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
                ax2.set_xticks(range(len(reward_names)))
                ax2.set_yticks(range(len(reward_names)))
                ax2.set_xticklabels(reward_names, rotation=45, ha="right")
                ax2.set_yticklabels(reward_names)
                ax2.set_title("Reward Correlation")
                plt.colorbar(im, ax=ax2, label="Correlation")
            except Exception as e:
                ax2.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
        else:
            ax2.text(0.5, 0.5, "Need >= 2 rewards", ha="center", va="center")

        # 3. 原始值趋势
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(data["raw"], label="Total Raw", linewidth=2)
        ax3.plot(data["log"], label="Total Log", linewidth=2)
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Value")
        ax3.set_title("Total Reward Trend")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 各奖励组件趋势
        ax4 = fig.add_subplot(gs[2, :])
        for name in reward_names:
            valid_data = [v for v in data[name] if not math.isnan(v)]
            ax4.plot(valid_data, label=name, alpha=0.7)
        ax4.set_xlabel("Step")
        ax4.set_ylabel("Value")
        ax4.set_title("Individual Reward Components")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Dashboard saved to {save_path}")
        else:
            plt.show()
        plt.close()


# ---------- 分阶段奖励管理 ----------
class Stage:
    """训练阶段定义

    每个阶段包含一组奖励组件，可以设置切换条件。
    """

    def __init__(
        self,
        name: str,
        episodes: Optional[int] = None,
        games: Optional[int] = None,
        condition: Optional[Callable[[], bool]] = None,
    ):
        self.name = name
        self.episodes = episodes  # 触发所需的 episode 数量
        self.games = games  # 触发所需的 game 数量
        self.condition = condition  # 自定义条件函数
        self._mgr = RewardMgr()  # 该阶段的奖励管理器

    def add(
        self,
        value: float,
        var: Optional[float] = None,
        max_var: float = 1.0,
        mul: float = 1.0,
        name: Optional[str] = None,
        clip: Optional[Union[float, tuple[float, float]]] = None,
    ) -> "Stage":
        """添加该阶段的奖励组件"""
        self._mgr.add(value, var, max_var, mul, name, clip)
        return self

    def get_reward(self) -> RewardMgr:
        """获取该阶段的奖励（返回新的 mgr）"""
        mgr = RewardMgr()
        for r in self._mgr._items:
            mgr._items.append(Reward(r.value, r.name))
        mgr._names = self._mgr._names.copy()
        return mgr

    def __repr__(self) -> str:
        conditions = []
        if self.episodes is not None:
            conditions.append(f"episodes>={self.episodes}")
        if self.games is not None:
            conditions.append(f"games>={self.games}")
        if self.condition is not None:
            conditions.append("custom_condition")
        return f"<Stage '{self.name}' {', '.join(conditions) if conditions else 'no_condition'}>"


class CurriculumMgr:
    """课程学习奖励管理器

    管理多个训练阶段，根据条件自动切换对应的奖励机制。
    """

    def __init__(self):
        self._stages: List[Stage] = []
        self._current_idx: int = 0
        self._episode_count: int = 0
        self._game_count: int = 0

    def add_stage(self, stage: Stage) -> "CurriculumMgr":
        """添加训练阶段（按添加顺序排序）"""
        self._stages.append(stage)
        return self

    def add_stages(self, *stages: Stage) -> "CurriculumMgr":
        """批量添加训练阶段"""
        for stage in stages:
            self.add_stage(stage)
        return self

    def get_current_stage(self) -> Optional[Stage]:
        """获取当前阶段"""
        if not self._stages:
            return None
        return self._stages[self._current_idx]

    def get_reward(self) -> RewardMgr:
        """获取当前阶段的奖励"""
        stage = self.get_current_stage()
        if stage is None:
            return RewardMgr()
        return stage.get_reward()

    def advance(
        self,
        episode_count: Optional[int] = None,
        game_count: Optional[int] = None,
        check_condition: bool = True,
    ) -> bool:
        """更新进度并检查是否需要切换阶段

        Args:
            episode_count: 当前 episode 数量
            game_count: 当前 game 数量
            check_condition: 是否检查自定义条件

        Returns:
            bool: 是否切换了阶段
        """
        if episode_count is not None:
            self._episode_count = episode_count
        if game_count is not None:
            self._game_count = game_count

        # 检查是否可以切换到下一个阶段
        next_idx = self._current_idx + 1
        if next_idx >= len(self._stages):
            return False  # 已经是最后一个阶段

        next_stage = self._stages[next_idx]

        # 检查条件
        should_advance = True
        if next_stage.episodes is not None and self._episode_count < next_stage.episodes:
            should_advance = False
        if next_stage.games is not None and self._game_count < next_stage.games:
            should_advance = False
        if check_condition and next_stage.condition is not None:
            if not next_stage.condition():
                should_advance = False

        if should_advance:
            self._current_idx = next_idx
            return True

        return False

    def reset(self) -> "CurriculumMgr":
        """重置到第一个阶段"""
        self._current_idx = 0
        self._episode_count = 0
        self._game_count = 0
        return self

    def __len__(self) -> int:
        return len(self._stages)

    def __repr__(self) -> str:
        current = self._stages[self._current_idx].name if self._stages else "None"
        return f"<CurriculumMgr stage={current}/{len(self)} ep={self._episode_count} game={self._game_count}>"


# 兼容性：导入 numpy 用于可视化
try:
    import numpy as np
except ImportError:
    pass
