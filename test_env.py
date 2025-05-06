from trading_env import StockTradingEnv
import pandas as pd

df = pd.read_csv("data/csi300_processed.csv").drop(columns=["date"])
env = StockTradingEnv(df)

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()