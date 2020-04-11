from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

# multiprocess environment
env = make_vec_env("CartPole-v1", n_envs=4)


model = PPO2(policy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo2_cartpole")

del model  # remove to demonstrate saving and loading

model = PPO2.load("ppo2_cartpole")

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
