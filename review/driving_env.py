import gym
from airsim import *

import numpy as np
import time
import random
from util import EnvInfo, MultiRotorEnvInfo
from numpy.linalg import norm

class MultiRotorEnv(gym.Env):
    def __init__(self, drone_id, wind=False, speed=1.0, ip='127.0.0.1', verbose=1, port=8000, seed=100, is_distort=True):
        self.drone_id = drone_id
        self.verbose = verbose
        self.is_distort = is_distort
        if self.verbose:
            print('Drone ID: ', self.drone_id)

        random.seed(seed)

        self.client = MultirotorClient(ip=ip, port=port)
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name = self.drone_id)
        self.client.armDisarm(True, vehicle_name = self.drone_id)
        self.client.takeoffAsync(vehicle_name = self.drone_id).join()

        self.multi_rotor_info = MultiRotorEnvInfo()

        self.info = EnvInfo()
        self.substep = 0
        self.timesteps = 0


        self._wind = wind
        self.theta = 0
        self.speed = speed
        self.temp_time = time.time()
        self.time_per_step = 0.02 / self.speed
        self.distortion = 0.0

        self.action_space = gym.spaces.Box(low=np.full((4, ), -1), high=np.full((4, ), 1), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.full(23, -1), high=np.full(23, -1), dtype=np.float32)

        self.prev_pwm = [0.5, 0.5, 0.5, 0.5]

        print(self.action_space.shape)
        print(self.observation_space.shape)

    def step(self, action: np.ndarray):
        already_col = False
        done = False
        self.timesteps += 1

        while time.time() - self.temp_time < self.time_per_step:
            time.sleep(min(max(self.time_per_step - (time.time() - self.temp_time), 0), 0.001))
            col_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_id)
            if col_info.has_collided:
                already_col = True
                break
        self.temp_time = time.time()

        #if self.is_distort:
        #    self.make_distortion()

        col_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_id)
        pos_info = self.client.simGetGroundTruthKinematics(vehicle_name=self.drone_id)
        imu_data = self.client.getImuData(imu_name='imu', vehicle_name=self.drone_id)
        gps_data = self.client.getGpsData(gps_name='gps', vehicle_name=self.drone_id)

        cur_pwm = [0, 0, 0, 0]
        for i in range(4):
            cur_pwm[i] = self.prev_pwm[i] + action[i] * 0.05
            if cur_pwm[i] > 1:
                cur_pwm[i] = 1
            elif cur_pwm[i] < 0:
                cur_pwm[i] = 0

        self.prev_pwm = cur_pwm

        self.client.moveByMotorPWMsAsync(float(cur_pwm[0]),
                                         float(cur_pwm[1]),
                                         float(cur_pwm[2]),
                                         float(cur_pwm[3]),
                                         1, vehicle_name=self.drone_id)

        self.wind_step()
        reward = self.multi_rotor_info.step(pos_info, imu_data, gps_data, action)
        self.info.step(reward)

        info = {'episode': None, 'success': False,
                'distance': self.multi_rotor_info.goal_distance, 'step':self.timesteps}

        if self.timesteps > 2000:
            done = True

        if col_info.has_collided or already_col :
            done = True
            _pos,_ = self.multi_rotor_info.get_sensor_data()
            print(_pos)

        state = self.multi_rotor_info.make_state()

        return state, reward, done, info

    def reset(self):
        if self.verbose:
            print(self.info)
            print(self.multi_rotor_info)

        self.timesteps = 0
        self.substep = 0
        self.client.reset()
        self.client.enableApiControl(True, vehicle_name=self.drone_id)
        self.client.armDisarm(True, vehicle_name=self.drone_id)
        self.client.takeoffAsync(vehicle_name=self.drone_id).join()

        self.prev_pwm = [0.5, 0.5, 0.5, 0.5]


        self.client.moveToPositionAsync(0, 0, -20, 10).join()
        time.sleep(4)

        col_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_id)
        pos_info = self.client.simGetGroundTruthKinematics(vehicle_name = self.drone_id)
        imu_data = self.client.getImuData(imu_name='imu', vehicle_name=self.drone_id)
        gps_data = self.client.getGpsData(gps_name='gps', vehicle_name=self.drone_id)

        self.multi_rotor_info.reset(pos_info, imu_data, gps_data)
        _pos,_ = self.multi_rotor_info.get_sensor_data()
        print(f'pos = {_pos}')
        print(f'goal = {self.multi_rotor_info.goal}')

        state = self.multi_rotor_info.make_state()

        self.temp_time = time.time()
        self.info.reset()
        return state

    def wind_step(self):
        if self._wind:
            wind = np.array([np.cos(self.theta), np.sin(self.theta)]) * np.random.random() * 20
            self.theta += np.pi / 180 * (np.random.normal(0, 1) * 10)

            wind = Vector3r(wind[0], wind[1], 0)
            self.client.simSetWind(wind)
        return

    def render(self, mode='human'):
        pass

    def make_distortion(self):
        if self.timesteps >= 10 :
            kinematics_state = self.client.simGetGroundTruthKinematics(vehicle_name=self.drone_id)

            dist = np.random.normal(0, self.distortion, size=(4,))
            orientation = kinematics_state.orientation
            orientation.x_val += dist[0]
            orientation.y_val += dist[1]
            orientation.z_val += dist[2]
            orientation.w_val += dist[3]
            #print("orientation {0}, {1}, {2}".format(orientation.x_val, orientation.y_val, orientation.z_val))

            dist = np.random.normal(0, self.distortion * 10, size=(3,))
            linear_acceleration = kinematics_state.linear_acceleration
            linear_acceleration.x_val += dist[0]
            linear_acceleration.y_val += dist[1]
            linear_acceleration.z_val += dist[2]
            #print("linear_acceleration{0}, {1}, {2}".format(linear_acceleration.x_val, linear_acceleration.y_val, linear_acceleration.z_val))

            dist = np.random.normal(0, self.distortion * 3.14, size=(3,))
            angular_velocity = kinematics_state.angular_velocity
            angular_velocity.x_val += dist[0]
            angular_velocity.y_val += dist[1]
            angular_velocity.z_val += dist[2]
            #print("angular_velocity{0}, {1}, {2}".format(angular_velocity.x_val, angular_velocity.y_val, angular_velocity.z_val))

            kinematics_state.orientation = orientation
            kinematics_state.linear_acceleration = linear_acceleration
            kinematics_state.angular_velocity = angular_velocity
            self.client.simSetKinematics(kinematics_state, True, self.drone_id)
