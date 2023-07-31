from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv

from driving_env import MultiRotorEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    save_name = 'Airsim_models'

    env = MultiRotorEnv(drone_id="Drone4", speed=2.7, ip='127.0.0.1', port=8003)
    model = SAC('MlpPolicy', env, verbose=1, buffer_size=500000)

    callback = CheckpointCallback(save_freq=50000, save_path=save_name)
    model.learn(total_timesteps=1000000, callback=callback, log_interval=10)
    model.save(save_name)
