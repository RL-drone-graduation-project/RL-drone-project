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
        self.imu_data = None
        self.gps_data = None

        self.prev_pwm = None
        self.rotor = None
        self.prev_pos_info = None    # 이전 상태정보가 들어갈 예정
        self.dt = 0.02    # delta time --> driving_env에서 self.time_per_step

    def __repr__(self):
        print("success: ", self.goal_reward)
        return " "

    def step(self, pos_info, imu_data, gps_data, last_action, prev_pwm, rotor):
        self.last_action = last_action
        self.prev_pwm = prev_pwm    # prev_pwm -> state에 추가
        self.rotor = rotor
        self.prev_pos_info = self.pos_info     # prev를 이전 self.pos_info로 설정
        self.pos_info = pos_info
        self.imu_data = imu_data
        self.gps_data = gps_data
        reward = self.get_reward()
        self.prev_distance = self.goal_distance
        return reward

    def reset(self, init_position, pos_info, imu_data, gps_data, prev_pwm, rotor):
        print("reset")
        self.goal = init_position
        self.prev_pos_info = pos_info    #전 상태도 똑같이 초기화
        self.pos_info = pos_info
        self.imu_data = imu_data
        self.gps_data = gps_data
        self.prev_distance = self.goal_distance
        self.goal_reward = 0
        self.last_action = np.array([0.5947, 0.5947, 0.5947, 0.5947])
        self.prev_pwm = prev_pwm
        self.rotor = rotor
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
        sen_pos, sen_vel = self.get_sensor_data()
        goal_data = self.goal

        pos = self.gps_to_position(geo_point)
        '''
        _pos = ', '.join(f'{x:.5f}' for x in np.nditer(pos))
        _sen_pos = ', '.join(f'{x:.5f}' for x in np.nditer(sen_pos))
        print(f'pos = {_pos}, sen_pos = {_sen_pos}')
        print(f'vel = {vel}, sen_vel = {sen_vel}')
        '''
        state = np.concatenate([ang_vel, lin_acc, ori, pos, vel, goal_data,
                                self.last_action])
        return state

    @property
    def goal_distance(self):
        return np.array([self.pos_info.position.x_val - self.goal[0], self.pos_info.position.y_val - self.goal[1],
                  self.pos_info.position.z_val - self.goal[2]])

    def get_reward(self):
        ori = np.array([self.pos_info.orientation.w_val,
                        self.pos_info.orientation.x_val,
                        self.pos_info.orientation.y_val,
                        self.pos_info.orientation.z_val])

        ret = 0
        ret -= np.clip(norm(self.goal_distance, 2) / 10, 0, 1)
        ang_vel = np.abs(np.array([self.pos_info.angular_velocity.x_val,
                                   self.pos_info.angular_velocity.y_val,
                                   self.pos_info.angular_velocity.z_val]))
        sum_ang = ang_vel[0] + ang_vel[1] + ang_vel[2]
        ret -= np.clip(sum_ang / np.pi, 0, 1)

        if norm(self.goal_distance,2) < 1 :
            ret += 2
        elif norm(self.goal_distance, 2) < 4:
            ret += 1

        return ret

    def get_sensor_data(self):
        position = np.array([self.pos_info.position.x_val,
                             self.pos_info.position.y_val,
                             self.pos_info.position.z_val])
        velocity = np.array([self.pos_info.linear_velocity.x_val,
                             self.pos_info.linear_velocity.y_val,
                             self.pos_info.linear_velocity.z_val])
        return position, velocity

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