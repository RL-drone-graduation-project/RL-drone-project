import gym
from airsim import *
import airsim
import math
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
        self.client.reset()
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
        self.time_per_step = 0.1 / self.speed
        self.step_num = 0

        gps_data = self.client.getGpsData(gps_name='gps', vehicle_name=self.drone_id)
        self.origin_position = np.array([47.641468, -122.140165, 122.932])

        self.client.moveToPositionAsync(0, 0, -20, 10).join()
        time.sleep(10)
        pos_info = self.client.simGetGroundTruthKinematics(vehicle_name=self.drone_id)
        gps_data = self.client.getGpsData(gps_name='gps', vehicle_name=self.drone_id)

        self.now_position = np.array([gps_data.gnss.geo_point.latitude,
                                       gps_data.gnss.geo_point.longitude,
                                       gps_data.gnss.geo_point.altitude])
        self.init_position = self.gps_to_position(self.now_position)

        print(f'now_position = {self.now_position}')
        print(f'origin_position = {self.origin_position}')
        print(f'init_position = {self.init_position}')

        self.distortion = 0.1
        self.episode_count = 0

        self.action_space = gym.spaces.Box(low=np.full((3, ), -1), high=np.full((3, ), 1), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.full(19, -1), high=np.full(19, -1), dtype=np.float32)
        print(self.action_space.shape)
        print(self.observation_space.shape)

    def gps_to_position(self, now_position):
        latitude = now_position[0]
        longitude = now_position[1]
        altitude = now_position[2]

        origin_latitude = 47.641468
        origin_longitude = -122.140165
        origin_altitude = 122.932

        a = 6378137.0  # m
        b = 6356752.314245  # m
        f = (a - b) / a

        lat_rad = math.radians(latitude)
        lon_rad = math.radians(longitude)
        alt = altitude

        e_squared = 2 * f - f ** 2
        N = a / math.sqrt(1 - e_squared * math.sin(lat_rad) ** 2)

        delta_latitude = latitude - origin_latitude
        delta_longitude = longitude - origin_longitude
        delta_altitude = altitude - origin_altitude

        x = (delta_longitude * math.pi / 180) * N * math.cos(lat_rad)
        y = (delta_latitude * math.pi / 180) * N
        z = -delta_altitude

        return np.array([y, x, z])

    def step(self, action: np.ndarray):
        already_col = False
        done = False
        self.timesteps += 1

        while time.time() - self.temp_time < self.time_per_step:
            time.sleep(min(max(self.time_per_step - (time.time() - self.temp_time), 0), 0.01))
            col_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_id)
            if col_info.has_collided:
                already_col = True
                break
        self.temp_time = time.time()

        if self.is_distort:
            self.make_distortion()

        col_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_id)
        pos_info = self.client.simGetGroundTruthKinematics(vehicle_name=self.drone_id)
        imu_data = self.client.getImuData(imu_name='imu', vehicle_name=self.drone_id)
        gps_data = self.client.getGpsData(gps_name='gps', vehicle_name=self.drone_id)


        self.client.moveByMotorPWMsAsync(0.5, 0.498, 0.498, 0.5, 0.5, vehicle_name=self.drone_id)

        if col_info.has_collided or already_col or self.timesteps > 1000:
            done = True
            _pos,_,_ = self.multi_rotor_info.get_sensor_data()
            print(_pos)
            print(self.timesteps)

        reward = self.multi_rotor_info.step(pos_info, imu_data, gps_data, action)
        self.info.step(reward)

        info = {'episode': None, 'success': False,
                'distance': norm(self.multi_rotor_info.goal_distance,2), 'step': self.timesteps,
                'zval': -pos_info.position.z_val}
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

        self.client.moveToPositionAsync(0, 0, -20, 10).join()

        self.step_num = 0

        time.sleep(5)

        col_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_id)
        pos_info = self.client.simGetGroundTruthKinematics(vehicle_name = self.drone_id)
        imu_data = self.client.getImuData(imu_name='imu', vehicle_name=self.drone_id)
        gps_data = self.client.getGpsData(gps_name='gps', vehicle_name=self.drone_id)

        self.multi_rotor_info.reset(self.origin_position, self.init_position, pos_info, imu_data, gps_data)

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
            print("orientation {0}, {1}, {2}".format(orientation.x_val, orientation.y_val, orientation.z_val))

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
