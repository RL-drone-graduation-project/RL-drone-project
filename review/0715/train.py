from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv

from driving_env import MultiRotorEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
#from stable_baselines3.common.vec_env import SubprocVecEnv
#from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3.common.vec_env import VecFrameStack

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

if __name__ == '__main__':
    save_name = 'train3_model'
    random_seed = 100

    env = MultiRotorEnv(drone_id="Drone3", speed=2.7, ip='127.0.0.1', port=8002, seed=random_seed)
    #env = DummyVecEnv([lambda: env])
    #env = VecFrameStack(env, n_stack=2)
    model = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        buffer_size=500000,
        learning_rate=0.0003,
        batch_size=512,
        gamma=0.95,
        tau=0.005,
        ent_coef='auto',
        target_entropy='auto',
        train_freq=1,
        gradient_steps=1,
        learning_starts=10000
    )

    callback = CheckpointCallback(save_freq=100000, save_path=save_name)
    model.learn(total_timesteps=2000000, callback=callback, log_interval=10)
    model.save(save_name)