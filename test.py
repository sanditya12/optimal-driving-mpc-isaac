from env import RacingEnv
from stable_baselines3 import PPO

policy_path = "./cnn_policy_271023/jetbot_policy"
# policy_path = "./cnn_policy/jetbot_policy_checkpoint_1000000_steps"

env = RacingEnv(headless=False, safety_filter=False)
model = PPO.load(policy_path)

for _ in range(20):
    obs = env.reset()
    done = False
    while not done:
        actions, _ = model.predict(observation = obs, deterministic = True)
        obs, reward, done, info = env.step(actions)
        env.render()

env.close()