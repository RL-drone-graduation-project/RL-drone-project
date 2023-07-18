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
        self.time_per_step = 0.07 / self.speed
        self.distortion = 0.0

        self.action_space = gym.spaces.Box(low=np.full((4, ), 0.25), high=np.full((4, ), 1), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.full(27, -1), high=np.full(27, -1), dtype=np.float32)

        self.prev_pwm = [0.5947, 0.5947, 0.5947, 0.5947]
        self.randxyz = None
        self.randxyz2 = None
        self.episode_num = 0

        print(self.action_space.shape)
        print(self.observation_space.shape)
        print(self.time_per_step)

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

        env_info = self.client.simGetGroundTruthEnvironment(vehicle_name=self.drone_id)
        col_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_id)
        pos_info = self.client.simGetGroundTruthKinematics(vehicle_name=self.drone_id)
        imu_data = self.client.getImuData(imu_name='imu', vehicle_name=self.drone_id)
        gps_data = self.client.getGpsData(gps_name='gps', vehicle_name=self.drone_id)

        if self.episode_num <=100 :
            #print(self.timesteps)
            if self.timesteps < 250 :
                self.client.moveToPositionAsync(self.randxyz[0], self.randxyz[1], self.randxyz[2]-20, 5)
            elif 250 <= self.timesteps < 500 :
                self.client.moveToPositionAsync(0, 0, -20, 5)
            elif 500 <= self.timesteps < 750 :
                self.client.moveToPositionAsync(self.randxyz2[0], self.randxyz2[1], self.randxyz2[2]-20, 5)
            else:
                self.client.moveToPositionAsync(0, 0, -20, 5)

            rotor = self.client.getRotorStates(vehicle_name=self.drone_id).rotors
            rotor_state = np.zeros((4,))
            for i in range(4):
                rotor_state[i] = rotor[i]['thrust'] #2.485475

            #print(rotor_state)
            pwm = self.thrust_to_pwm(rotor_state, env_info)
            _pos, _ = self.multi_rotor_info.get_sensor_data()
            #print(f'pwm = {pwm}, pos={_pos}')

            action = pwm

        else :
            #action = [0.5947,0.5947,0.5947,0.5947]

            self.client.moveByMotorPWMsAsync(float(action[0]),
                                             float(action[1]),
                                             float(action[2]),
                                             float(action[3]),
                                             1, vehicle_name=self.drone_id)

            rotor = self.client.getRotorStates(vehicle_name=self.drone_id).rotors
            rotor_state = np.zeros((4,))
            for i in range(4) :
                rotor_state[i] = rotor[i]['thrust']

            #print(f'rotor = {rotor_state}') #2.48551583
            if self.timesteps % 20 == 0 :
                print(action)
        cur_pwm = action

        self.wind_step()
        reward = self.multi_rotor_info.step(pos_info, imu_data, gps_data, action, cur_pwm, rotor_state)

        self.info.step(reward)

        info = {'episode': None, 'success': False,
                'distance': self.multi_rotor_info.goal_distance, 'step':self.timesteps,
                'new_action':action}

        if self.timesteps >= 1000:
            done = True
            _pos, _ = self.multi_rotor_info.get_sensor_data()
            print(_pos)

        if col_info.has_collided or already_col :
            done = True
            _pos,_ = self.multi_rotor_info.get_sensor_data()
            print(_pos)

        state = self.multi_rotor_info.make_state()

        #print(reward)
        #print(state)

        return state, reward, done, info

    def thrust_to_pwm(self, thrust, env):

        propeller_diameter = 0.2286
        standard_air_density = 1.2085

        max_thrust = 4.17944479
        air_density_ratio = env.air_density / standard_air_density
        #print(f'thrust = {thrust}')
        pwm = np.zeros_like(thrust)

        for i in range(thrust.size):
            pwm[i] = np.clip(thrust[i] / (air_density_ratio * max_thrust), 0.0, 1.0)

        return pwm

    def generate_random_vector(self):
        while True:
            x = random.uniform(-10, 10)
            y = random.uniform(-10, 10)
            z = random.uniform(-10, 10)

            if x ** 2 + y ** 2 + z ** 2 <= 100:
                return x,y,z

    def reset(self):
        if self.verbose:
            print(self.info)
            print(self.multi_rotor_info)

        self.episode_num += 1

        self.timesteps = 0
        self.substep = 0
        self.client.reset()
        self.client.enableApiControl(True, vehicle_name=self.drone_id)
        self.client.armDisarm(True, vehicle_name=self.drone_id)
        self.client.takeoffAsync(vehicle_name=self.drone_id).join()

        self.prev_pwm = [0.5947, 0.5947, 0.5947, 0.5947]

        self.client.moveToPositionAsync(0, 0, -20, 10).join()
        time.sleep(4)

        env_info = self.client.simGetGroundTruthEnvironment(vehicle_name=self.drone_id)
        col_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_id)
        pos_info = self.client.simGetGroundTruthKinematics(vehicle_name = self.drone_id)
        imu_data = self.client.getImuData(imu_name='imu', vehicle_name=self.drone_id)
        gps_data = self.client.getGpsData(gps_name='gps', vehicle_name=self.drone_id)

        rotor = self.client.getRotorStates(vehicle_name=self.drone_id).rotors   # rotor state 추가
        rotor_state = np.zeros((4,))
        for i in range(4):
            rotor_state[i] = rotor[i]['thrust']

        x_val, y_val, z_val = self.generate_random_vector()
        self.randxyz = np.array([x_val,y_val,z_val])
        x_val2, y_val2, z_val2 = self.generate_random_vector()
        self.randxyz2 = np.array([x_val2, y_val2, z_val2])


        if self.episode_num % 10 == 0 :
            self.randxyz = np.zeros((3,))
            self.randxyz2 = np.zeros((3,))

        print(f'rand = {self.randxyz}')
        print(f'rand2 = {self.randxyz2}')


        self.multi_rotor_info.reset(pos_info, imu_data, gps_data, self.prev_pwm, rotor_state)
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
    '''
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
        '''
