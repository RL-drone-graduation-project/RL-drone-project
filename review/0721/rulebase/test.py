from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv

from driving_env import MultiRotorEnv
from stable_baselines3 import SAC

#from callback import CustomCallback

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt

def plotting(x, y, str):
    plt.clf()
    plt.plot(x, y)
    plt.xlabel('Distortion rate')
    plt.ylabel(str)
    plt.title(str + " Graph")
    plt.grid(True)

    plt.savefig(str + ' graph.png')
    plt.show()


if __name__ == '__main__':

    save_name = 'vel_distortion.zip'
    random_seed = 100

    env = MultiRotorEnv(drone_id="Drone2", speed=2.7, ip='127.0.0.1', port=8001, seed=random_seed)
    model = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        buffer_size=200000,
        learning_rate=0.0001,
        batch_size=512,
        gamma=0.95,
        tau=0.005,
        ent_coef='auto',
        target_entropy='auto',
        train_freq=1,
        gradient_steps=1,
        seed=random_seed
    )
    model = SAC.load(save_name)


    mean_distance_avg = []

    for j in range(15):
        minzlist = []
        falltime = []
        mean_distance = []

        for k in range(5) :
            obs = env.reset()
            dones = False
            ep_reward = 0

            steplist = list()
            distlist = list()
            rewlist = list()
            zlist = []
            num = 0
            for i in range(200) :
                num=i
                if dones :
                    break
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                #print(f'step = {info["step"]}, dist={info["distance"]}, rew = {rewards}')
                steplist.append(info["step"])
                distlist.append(info["distance"])
                zlist.append(info["zval"])
                ep_reward += rewards
                rewlist.append(ep_reward)

            falltime.append(num)
            minzlist.append(min(zlist))

            print(f'ep_reward = {ep_reward}')
            print(f'mean_distance = {sum(distlist) / len(distlist)}')
            print(f'zval = {info["zval"]}')
            mean_distance.append(sum(distlist) / len(distlist))

        mean_distance_avg.append(sum(mean_distance) / 5)
        print(f'mean mean distance = {sum(mean_distance) / 5}')
        #plotting(steplist, distlist, str="Distance")
        #plotting(steplist, rewlist, str="Reward")

    #plotting([2*i for i in range(1,16)], falltime, str="Fallen time")
    plotting([2 * i for i in range(1, 16)], mean_distance_avg, str="Avg Distance of 5 Episodes")
