import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

class StockTradingEnv(gym.Env):
    """股票交易强化学习环境(Gymnasium兼容版)"""
    
    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 df: pd.DataFrame,
                 initial_balance: float = 100000,
                 transaction_cost: float = 0.0003,
                 max_shares: int = 10000,
                 render_mode: str = None):
        super().__init__()
        
        # 数据验证和处理
        self.df = self._preprocess_data(df.copy())
        self._validate_data()
        
        # 环境参数
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_shares = max_shares
        self.render_mode = render_mode
        
        # 观察空间和动作空间
        self.n_features = self.df.shape[1]
        self.account_features = 2
        self.obs_dim = self.n_features + self.account_features
        
        self.observation_space = spaces.Box(
            low=-10,  # 设置合理下限
            high=10,  # 设置合理上限
            shape=(self.obs_dim,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(3)
        
        # 状态变量
        self.reset()

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        df = df.astype(np.float32)
        # 添加微小噪声防止零除
        df += np.random.normal(0, 1e-6, size=df.shape)
        return df

    def _validate_data(self) -> None:
        """验证数据完整性"""
        if self.df.isnull().values.any():
            raise ValueError("数据包含NaN值")
        if np.isinf(self.df.values).any():
            raise ValueError("数据包含无穷值")

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """重置环境状态"""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.portfolio_value = self.initial_balance
        self.trades = 0
        
        observation = self._next_observation()
        info = self._get_info()
        
        return observation, info

    def _next_observation(self) -> np.ndarray:
        """获取下一个观察值"""
        market_data = self.df.iloc[self.current_step].values
        account_info = np.array([
            self.shares_held / (self.max_shares + 1e-6),
            np.clip(self.balance / (self.initial_balance + 1e-6), 0, 2)
        ], dtype=np.float32)
        
        obs = np.concatenate([market_data, account_info])
        
        # 数值检查和处理
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = np.clip(obs, -10, 10)
        
        return obs

    def _get_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        return {
            "step": self.current_step,
            "balance": self.balance,
            "shares": self.shares_held,
            "portfolio_value": self.portfolio_value,
            "trades": self.trades
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步动作"""
        if self.current_step >= len(self.df) - 1:
            terminated = True
            truncated = False
            return self._next_observation(), 0, terminated, truncated, self._get_info()
            
        current_price = self.df.iloc[self.current_step]["close"]
        prev_value = max(self.portfolio_value, 1e-6)  # 防止零除
        
        # 执行交易
        self._execute_trade(action, current_price)
        
        # 更新状态
        self.current_step += 1
        new_price = self.df.iloc[self.current_step]["close"]
        self.portfolio_value = self.balance + self.shares_held * new_price
        
        # 计算奖励
        reward = self._calculate_reward(prev_value)
        
        # 检查是否结束
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        if self.render_mode == "human":
            self.render()
        
        return self._next_observation(), reward, terminated, truncated, self._get_info()

    def _execute_trade(self, action: int, current_price: float) -> None:
        """执行交易逻辑"""
        if action == 1 and current_price > 1e-6:  # 买入
            max_buy = (self.balance * 0.2) / (current_price * (1 + self.transaction_cost))
            shares_bought = min(int(max_buy // 100) * 100, 10000)
            cost = shares_bought * current_price * (1 + self.transaction_cost)
            
            if cost > 0 and cost <= self.balance:
                self.balance -= cost
                self.shares_held += shares_bought
                self.trades += 1
                
        elif action == 2 and self.shares_held > 0:  # 卖出
            shares_sold = min(int(self.shares_held * 0.2), self.shares_held)
            revenue = shares_sold * current_price * (1 - self.transaction_cost)
            
            if shares_sold > 0:
                self.balance += revenue
                self.shares_held -= shares_sold
                self.trades += 1

    def _calculate_reward(self, prev_value: float) -> float:
        """计算奖励"""
        if prev_value <= 1e-6 or self.portfolio_value <= 1e-6:
            return 0
            
        # 对数收益率
        ret = np.log(max(self.portfolio_value, 1e-6) / max(prev_value, 1e-6))
        
        # 限制奖励范围
        return np.clip(ret, -1, 1)

    def render(self) -> None:
        """渲染当前状态"""
        current_price = self.df.iloc[self.current_step]["close"]
        print(
            f"Step: {self.current_step} | "
            f"Price: {current_price:.2f} | "
            f"Shares: {self.shares_held} | "
            f"Balance: {self.balance:.2f} | "
            f"Value: {self.portfolio_value:.2f}"
        )

    def close(self):
        pass