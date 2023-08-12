from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import LnMlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv

from driving_env import MultiRotorEnv
from stable_baselines import SAC
from stable_baselines.common.callbacks import CheckpointCallback
#from stable_baselines3.common.vec_env import SubprocVecEnv
#from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines.common.buffers import ReplayBuffer

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt


def plotting(x, y, str):
    plt.clf()
    plt.plot(x, y)
    plt.xlabel('Steps')
    plt.ylabel(str)
    plt.title(str + " Graph")
    plt.grid(True)

    plt.savefig(str + 'graph.png')
    plt.show()


if __name__ == '__main__':

    save_name = 'rl_model_1000000_steps.zip'
    random_seed = 100


    env = MultiRotorEnv(drone_id="Drone2", speed=1.0, ip='127.0.0.1', port=8001, seed=random_seed)
    '''
    model = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        buffer_size=1000000,
        learning_rate=0.0003,
        batch_size=512,
        gamma=0.95,
        tau=0.005,
        ent_coef='auto',
        target_entropy='auto',
        train_freq=1,
        gradient_steps=1, 
        learning_starts=100
    )'''
    model = SAC.load(save_name)

    for _ in range(4):
        obs = env.reset()
        dones = False
        ep_reward = 0

        steplist = list()
        distlist = list()
        rewlist = list()

        for i in range(1000):
            if dones:
                break
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            # print(f'step = {info["step"]}, dist={info["distance"]}, rew = {rewards}')
            #print(f'step = {i}')
            steplist.append(info["step"])
            distlist.append(info["distance"])
            ep_reward += rewards
            rewlist.append(ep_reward)

        #plotting(steplist, distlist, str="Distance")
        #plotting(steplist, rewlist, str="Reward")
        #print(f'Mean distance = {sum(distlist) / len(distlist)}')
        print(f'goal_distance = {info["distance"]}')
        print(f'ep_reward = {ep_reward}')


