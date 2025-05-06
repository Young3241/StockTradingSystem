import os
import numpy as np
import torch
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from trading_env import StockTradingEnv
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"ppo_model_{self.n_calls}")
            self.model.save(model_path)
        return True

class Config:
    TRAIN_DATA_PATH = "data/csi300_processed.csv"
    MODEL_SAVE_DIR = "models"
    LOG_DIR = "logs"
    TOTAL_TIMESTEPS = 100000
    SAVE_FREQ = 10000
    
    SCALER_COLS = [
        'open', 'high', 'low', 'close', 
        'volume', 'MA5', 'MA20', 'RSI'
    ]
    
    POLICY_KWARGS = dict(
        activation_fn=torch.nn.LeakyReLU,
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )
    
    LEARNING_RATE = 1e-4
    N_STEPS = 2048
    BATCH_SIZE = 64
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    MAX_GRAD_NORM = 0.5
    CLIP_RANGE = 0.2
    ENT_COEF = 0.01

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """加载并预处理数据"""
    df = pd.read_csv(filepath)
    
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    # 填充缺失值
    df = df.ffill().bfill()
    
    # 标准化特征
    scaler = StandardScaler()
    df[Config.SCALER_COLS] = scaler.fit_transform(df[Config.SCALER_COLS])
    
    # 添加微小噪声
    df += np.random.normal(0, 1e-6, size=df.shape)
    
    print("\n=== 数据预处理结果 ===")
    print("NaN值数量:", df.isnull().sum().sum())
    print("无穷值数量:", np.isinf(df.values).sum())
    print("数据形状:", df.shape)
    
    return df

def create_env(df: pd.DataFrame) -> DummyVecEnv:
    """创建训练环境"""
    env = StockTradingEnv(df)
    env = Monitor(env)
    return DummyVecEnv([lambda: env])

def setup_model(env: DummyVecEnv) -> PPO:
    """设置PPO模型"""
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=Config.LEARNING_RATE,
        n_steps=Config.N_STEPS,
        batch_size=Config.BATCH_SIZE,
        gamma=Config.GAMMA,
        gae_lambda=Config.GAE_LAMBDA,
        max_grad_norm=Config.MAX_GRAD_NORM,
        clip_range=Config.CLIP_RANGE,
        ent_coef=Config.ENT_COEF,
        policy_kwargs=Config.POLICY_KWARGS,
        verbose=1,
        tensorboard_log=Config.LOG_DIR,
        device="auto"
    )

def train_ppo() -> None:
    """训练PPO模型"""
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 准备数据
    df = load_and_preprocess_data(Config.TRAIN_DATA_PATH)
    
    # 创建环境
    env = create_env(df)
    
    # 创建模型
    model = setup_model(env)
    
    # 创建输出目录
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # 训练模型
    try:
        model.learn(
            total_timesteps=Config.TOTAL_TIMESTEPS,
            callback=SaveModelCallback(
                save_freq=Config.SAVE_FREQ,
                save_path=Config.MODEL_SAVE_DIR
            ),
            tb_log_name="PPO"
        )
    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        raise
    
    # 保存最终模型
    model.save(os.path.join(Config.MODEL_SAVE_DIR, "ppo_final"))
    print("训练完成!")

if __name__ == "__main__":
    train_ppo()