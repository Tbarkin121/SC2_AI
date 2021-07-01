import gym
from sc2_gym.envs import DZBEnv

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])
new_agent = True
# create vectorized environment
# env = gym.make('defeat-zerglings-banelings-v0')
# eng = DZBEnv()
# env = DummyVecEnv([lambda: DZBEnv()])
# env = VecNormalize(env, norm_obs=True, norm_reward=True,
#                    clip_obs=10.)
env = make_vec_env('defeat-zerglings-banelings-v0', n_envs=4, seed=0)

# use ppo2 to learn and save the model when finished
if(new_agent=True):
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="log/")
else:
    # Load the agent
    model = PPO.load("model/dbz_ppo", env=env)

model.learn(total_timesteps=int(1e6), tb_log_name="first_run", reset_num_timesteps=False)
model.save("model/dbz_ppo")

