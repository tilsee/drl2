import numpy as np

from src.sb3.stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from src.sb3.stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs
from collections import deque

class VecBALLSAVERWrapper(VecEnvWrapper):

    def __init__(self, venv: VecEnv, gamma: float = 0.99):
        VecEnvWrapper.__init__(self, venv, venv.observation_space, venv.action_space)
        self.venv = venv
        self.gamma = gamma
        self.history = []#deque(maxlen=5)  # History of last 5 state infos
        self.current_round_steps = 0
    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.history.clear()  # Clear history on reset

        return obs

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()
        if infos[0]["black_conceded"] == 1 or infos[0]["white_conceded"] == 1:
            self.current_round_steps = 0
        self.current_round_steps += 1
        x_vel = infos[0]["ball_x_velocity"]
        reward_ball_vel = min(0.4,x_vel/abs(x_vel)*x_vel*2)
        time_penality_ratio = 0.001
        penalty_time = time_penality_ratio*self.current_round_steps
        rewards +=  reward_ball_vel + penalty_time
        return observations, rewards, dones, infos

def contour_plot_to_console(X, Z, Y):
    for i in range(len(X)):
        if i % 5 == 0:
            row = ""
            for j in range(len(Z)):
                if j % 5 == 0:
                    row += " " + f"{Y[i][j]:.2f}"
            print(row)


def calculate_intercept(ball_x_pos, ball_y_pos, ball_x_vel, ball_y_vel, goalie_x_line=-1.0573):
    if ball_x_vel < 0:  # intercept only possible if ball moving towards goalie
        t_intercept = (goalie_x_line - ball_x_pos) / ball_x_vel
        y_intercept = ball_y_pos + ball_y_vel * t_intercept
        return y_intercept, t_intercept
    return None, None


'''
        info = {
            "black_conceded": 1 if black_conceded else 0,
            "white_conceded": 1 if white_conceded else 0,
            "ball_x_position": ball_pos[0], # ranges from -1.5 to 1.5
            "ball_y_position": ball_pos[1], # hight of the ball over the field when doing airtime
            "ball_z_position": ball_pos[2], # ranges from -0.5 to 0.5
            "ball_x_velocity": self.data.qvel[2], # x component of the ball velocity
            "ball_z_velocity": self.data.qvel[3], # z component of the ball velocity
            "goalie_z_position": self.data.qvel[0], # z component of the goalie position
            "goalie_angular_pos": self.data.qvel[1] # rotation of the goalie [-0.56, 0.56]
        }
'''


def calculate_reward(info, previous_info):
    reward = 0
    if info['black_conceded']:
        reward -= 1
    if info['white_conceded']:
        reward -= 1
    
    # Encourage positional accuracy
    position_diff = abs(info['ball_z_position'] - info['goalie_z_position'])
    if position_diff < 0.05:  # threshold for "good positioning"
        reward += 0.1

    # Discourage unnecessary movements
    movement_penalty = 0.01 * abs(info['goalie_z_position'] - previous_info['goalie_z_position'])
    reward -= movement_penalty

    # Reward for moving the ball towards the opponent's goal
    if info['ball_x_velocity'] > 0:
        reward += 0.05

    return reward


# goaly rotates too much
# def calculate_reward(info):
#     reward = 0

#     # Constants for scaling rewards
#     GOAL_REWARD = 10
#     CONCEDE_PENALTY = -10
#     INTERCEPT_REWARD = 5
#     DISTANCE_PENALTY_SCALE = 1

#     # Reward for scoring a goal
#     if info["white_conceded"]:
#         reward += GOAL_REWARD

#     # Penalty for conceding a goal
#     if info["black_conceded"]:
#         reward += CONCEDE_PENALTY

#     # Calculate distance between the goalie and the ball
#     distance_to_ball = abs(info["goalie_z_position"] - info["ball_z_position"])

#     # Reward for intercepting the ball (when distance is minimal)
#     if distance_to_ball < 0.05:  # Threshold for interception, can be adjusted
#         reward += INTERCEPT_REWARD

#     # Reward for redirecting the ball towards the opponent's goal
#     if info["ball_x_velocity"] > 0 and distance_to_ball < 0.05:
#         reward += info["ball_x_velocity"]

#     # Penalty based on the distance to the ball
#     reward -= DISTANCE_PENALTY_SCALE * distance_to_ball

#     return reward
