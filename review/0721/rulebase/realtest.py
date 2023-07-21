from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv

from driving_env import MultiRotorEnv
from stable_baselines3 import SAC

# from callback import CustomCallback

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt


def plotting(step, x, y, z, str, ylim):
    plt.clf()
    plt.plot(step, x, 'r', label="X")
    plt.plot(step, y, 'g', label="Y")
    plt.plot(step, z, 'b', label="Z")
    plt.xlabel('Steps')
    plt.ylabel(str)
    plt.ylim(ylim)
    plt.title(str + " Graph")
    plt.grid(True)
    plt.legend()

    plt.savefig(str + 'graph.png')
    #plt.show()


if __name__ == '__main__':

    save_name = 'vel_distortion.zip'
    random_seed = 100

    env = MultiRotorEnv(drone_id="Drone2", speed=2.7, ip='127.0.0.1', port=8001, seed=random_seed)
    model = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        buffer_size=200000,
        learning_rate=0.0003,
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

    for _ in range(1):
        obs = env.reset()
        dones = False
        ep_reward = 0

        steplist = list()
        distlist = list()
        rewlist = list()


        angx = list()
        angy = list()
        angz = list()
        linx = list()
        liny = list()
        linz = list()

        for i in range(300):
            if dones:
                break
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            # print(f'step = {info["step"]}, dist={info["distance"]}, rew = {rewards}')
            steplist.append(info["step"])
            #distlist.append(info["distance"])
            ep_reward += rewards
            #rewlist.append(ep_reward)
            angx.append(info['ang'].x_val)
            angy.append(info['ang'].y_val)
            angz.append(info['ang'].z_val)
            linx.append(info['lin'].x_val)
            liny.append(info['lin'].y_val)
            linz.append(info['lin'].z_val)

        plotting(steplist, angx, angy, angz, "Angular Velocity without noise",[-1.5,1.5])

        plotting(steplist, linx, liny,linz,"Linear Acceleration without noise", [-13.5,4])
        #plotting(steplist, distlist, str="Distance")
        #plotting(steplist, rewlist, str="Reward")
        #print(f'Mean distance = {sum(distlist) / len(distlist)}')

        print(f'ep_reward = {ep_reward}')


