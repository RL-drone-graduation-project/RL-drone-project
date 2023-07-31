import time
import numpy as np
from numpy.linalg import norm
from util.spot import spot_list
from util.util import vector_process
import math


class EnvInfo(object):
    def __init__(self):
        self.episode_length = 0
        self.episode_reward = 0
        self.total_steps = 0
        self.start_time = time.time()

    def __repr__(self):
        print("episode length: ", self.episode_length)
        print("episode reward: ", self.episode_reward)
        print("total timesteps: ", self.total_steps)
        print("FPS: ", self.fps)
        return " "

    def step(self, reward):
        self.episode_length += 1
        self.episode_reward += reward
        self.total_steps += 1

    def reset(self):
        self.episode_length = 0
        self.episode_reward = 0
        self.start_time = time.time()

    @property
    def fps(self):
        return self.episode_length / ((time.time() -self.start_time) + 1e-10)


class MultiRotorEnvInfo(object):
    def __init__(self):
        self.goal_reward = 0
        self.prev_distance = None
        self.last_action = None
        self.goal = None
        self.pos_info = None
        self.lidar_data = None
        self.imu_data = None
        self.gps_data = None
        self.origin_position = None

    def __repr__(self):
        print("success: ", self.goal_reward)
        return " "

    def step(self, pos_info, imu_data, gps_data, last_action):
        self.last_action = last_action
        self.pos_info = pos_info
        self.imu_data = imu_data
        self.gps_data = gps_data
        reward = self.get_reward()
        self.prev_distance = self.goal_distance
        return reward

    def reset(self, origin_position, init_position, pos_info, imu_data, gps_data):
        print("reset")

        self.origin_position = origin_position   # 시작 gps 좌표
        self.goal = init_position                      # goal
        print(f'goal = {self.goal}')
        self.pos_info = pos_info
        self.imu_data = imu_data
        self.gps_data = gps_data
        self.prev_distance = self.goal_distance
        self.goal_reward = 0
        self.last_action = np.zeros((3,))
        return

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

    def make_state(self):
        ang_vel, lin_acc, ori = self.get_imu_data()
        geo_point, vel = self.get_gps_data()
        sen_pos, _,_ = self.get_sensor_data()
        goal_data = self.goal

        pos = self.gps_to_position(geo_point)
        print(f'pos = {pos}, sen_pos = {sen_pos}')
        state = np.concatenate([ang_vel, lin_acc, ori, sen_pos, vel, goal_data])
        return state

    @property
    def goal_distance(self):
        return np.array([self.pos_info.position.x_val - self.goal[0], self.pos_info.position.y_val - self.goal[1],
                  self.pos_info.position.z_val - self.goal[2]])

    def get_reward(self):
        if norm(self.goal_distance, 2) < 2:
            return 1
        else:
            return 0

    def get_sensor_data(self):
        position = np.array([self.pos_info.position.x_val,
                             self.pos_info.position.y_val,
                             self.pos_info.position.z_val])
        velocity = np.array([self.pos_info.linear_velocity.x_val,
                             self.pos_info.linear_velocity.y_val,
                             self.pos_info.linear_velocity.z_val])
        orientation = np.array([self.pos_info.orientation.w_val,
                                self.pos_info.orientation.x_val,
                                self.pos_info.orientation.y_val,
                                self.pos_info.orientation.z_val])
        return position, velocity, orientation

    def get_imu_data(self):
        angular_velocity = np.array([self.imu_data.angular_velocity.x_val,
                                     self.imu_data.angular_velocity.y_val,
                                     self.imu_data.angular_velocity.z_val])
        linear_acceleration = np.array([self.imu_data.linear_acceleration.x_val,
                                        self.imu_data.linear_acceleration.y_val,
                                        self.imu_data.linear_acceleration.z_val])
        orientation = np.array([self.imu_data.orientation.w_val,
                                self.imu_data.orientation.x_val,
                                self.imu_data.orientation.y_val,
                                self.imu_data.orientation.z_val])
        return angular_velocity, linear_acceleration, orientation

    def get_gps_data(self):
        geo_point = np.array([self.gps_data.gnss.geo_point.latitude,
                              self.gps_data.gnss.geo_point.longitude,
                              self.gps_data.gnss.geo_point.altitude])
        velocity = np.array([self.gps_data.gnss.velocity.x_val,
                             self.gps_data.gnss.velocity.y_val,
                             self.gps_data.gnss.velocity.z_val])
        return geo_point, velocity


'''
    def get_lidar_data(self):
        result = np.ones(shape=(16, 8)) * 30
        points = self.lidar_data.point_cloud
        if len(points) == 1:
            return result
        points = np.array(points).reshape([-1, 3])
        for vector in points:
            distance, theta, phi = vector_process(vector, self.pos_info)
            y_axis = int(theta/(np.pi/8))
            x_axis = int(phi/(np.pi/8))
            result[x_axis][y_axis] = min(distance, result[x_axis][y_axis])
        return result.flatten()
'''