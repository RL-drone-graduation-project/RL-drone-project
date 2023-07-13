import time
import numpy as np
from numpy.linalg import norm
from util.spot import spot_list
from util.util import vector_process

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
        self.dt = 0.02 / 2.7    # delta time --> driving_env에서 self.time_per_step

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

    def reset(self, pos_info, imu_data, gps_data, prev_pwm, rotor):
        print("reset")
        self.goal = np.array([0,0,-20])

        self.prev_pos_info = pos_info    #전 상태도 똑같이 초기화
        self.pos_info = pos_info
        self.imu_data = imu_data
        self.gps_data = gps_data
        self.prev_distance = self.goal_distance
        self.goal_reward = 0
        self.last_action = np.zeros((4,))
        self.prev_pwm = prev_pwm
        self.rotor = rotor

        return

    def make_state(self):
        ang_vel, lin_acc, ori = self.get_imu_data()
        geo_point, vel = self.get_gps_data()
        sen_pos, sen_vel = self.get_sensor_data()
        goal_data = self.goal
        state = np.concatenate([ang_vel, lin_acc, ori, sen_pos, sen_vel, goal_data,
                                self.last_action, self.prev_pwm, self.rotor])
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

        ret = 0.0
        ret += norm(self.prev_distance, 2) - norm(self.goal_distance, 2)


        ang_vel = np.array([self.pos_info.angular_velocity.x_val,
                            self.pos_info.angular_velocity.y_val,
                            self.pos_info.angular_velocity.z_val])
        prev_ang_vel = np.array([self.prev_pos_info.angular_velocity.x_val,
                                 self.prev_pos_info.angular_velocity.y_val,
                                 self.prev_pos_info.angular_velocity.z_val])

        target = np.abs(prev_ang_vel - ang_vel)
        sum_target = 0
        for i in range(3):
            sum_target += target[i]

        ret -= (sum_target/ (3*np.pi))

        '''
        ori = np.abs(ori - np.array([1.,0.,0.,0.])) # w 시작값은 1이기 때문
        for i in range(4) :
            if ori[i] < 0.4 :
                ret -= (np.exp(ori[i] / 5) - 1)
            else :
                ret -= (np.power(ori[i] * 2.5, 2))
        '''
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
        geo_point = np.array([self.gps_data.gnss.geo_point.altitude,
                              self.gps_data.gnss.geo_point.latitude,
                              self.gps_data.gnss.geo_point.longitude])
        velocity = np.array([self.gps_data.gnss.velocity.x_val,
                             self.gps_data.gnss.velocity.y_val,
                             self.gps_data.gnss.velocity.z_val])
        return geo_point, velocity