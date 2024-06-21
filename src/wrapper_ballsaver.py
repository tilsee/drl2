import numpy as np

from src.sb3.stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from src.sb3.stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs
from collections import deque
import pprint
from scipy.spatial import distance


class VecBALLSAVERWrapper(VecEnvWrapper):

    def __init__(self, venv: VecEnv, gamma: float = 0.99):
        VecEnvWrapper.__init__(self, venv, venv.observation_space, venv.action_space)
        self.venv = venv
        self.gamma = gamma
        self.current_round_steps = 0
        self.history = deque(maxlen=5)  # History of last 5 state infos

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        # Differenz zwischen 2 Hitory Zuständen nutzen 
        self.history.clear()  # Clear history on reset

        return obs

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()

        # Incentive for not moving the goalie
        #if len(self.history) > 0:
         #   first_info = self.history[0]  # Get the earliest info available
          #  dif_y = abs(infos[0]['goalie_y_position'] - first_info['goalie_y_position'])
           # dif_rot = abs(infos[0]['goalie_angle'] - first_info['goalie_angle'])
            #reward_rot = -0.5 * dif_rot
            #reward_zpos = -1 * dif_y
        #else:
         #   reward_rot = reward_zpos = 0

        # incentive for keeping figure down
        #reward_stay_down = 2 if abs(infos[0]['goalie_angle']) < 0.2 else -2
        #print(observations.shape)
        #print(observations)
        # incentive for moving towards ball intercept line. the closer the better
        #if infos[0]["ball_x_velocity"] < 0:
         #   y_intercept, time = calculate_intercept(infos[0]["ball_x_position"], infos[0]["ball_y_position"], infos[0]["ball_x_velocity"], infos[0]["ball_z_velocity"])
          #  goalie_dif_intercept = abs(y_intercept - infos[0]['goalie_y_position'])
           # reward_stay_center = -goalie_dif_intercept
            #print(reward_rot, reward_zpos, reward_stay_center, reward_stay_down, time, rewards)
            #rewards += reward_rot + reward_zpos + 0.5 * reward_stay_center + reward_stay_down
            #print(rewards)
        #else:
            #rewards += 0.1
            #print(rewards)
        timeout = infos[0]["TimeLimit.truncated"]
        black_conceded, white_conceded = infos[0]["black_conceded"], infos[0]["white_conceded"]
        if black_conceded or white_conceded:
            self.current_round_steps = 0
        self.current_round_steps += 1
        ball_x_velocity = infos[0]["ball_x_velocity"]
        #sign = np.sign(ball_x_velocity)
        #shot_value = sign * (2 * ball_x_velocity)
        ball_x, ball_y, ball_z = infos[0]["ball_position"]
        goal_distance = distance.euclidean((1.2, 0, 0.6), (ball_x, ball_y, ball_z))

        #if shot_value > 2:
         #   rewards -= 0.75
        #if shot_value > 1:
        #    rewards -= 0.25
        #if shot_value <= 1:
        #    rewards += 0.1
        #if shot_value <= 0.5:
        #    rewards += 1.5
        #if shot_value <= 0.25:
        #    rewards += 2
        #penalty_time = 0.001 * self.current_round_steps
        rewards -= goal_distance
        #rewards += self.gamma * ball_x_velocity
        #rewards -= penalty_time
        #rewards -= 2 * shot_value
        #rewards += 0.5 * ball_x_velocity
        #goal_distance = 2 * distance.euclidean((1.2, 0), (ball_x, ball_y))
        #rewards -= goal_distance
        #print(infos, goal_distance)

        print(rewards, goal_distance, (ball_x, ball_y, ball_z))
        # Update history with the current info
        self.history.append(infos[0])
        # disc. neues potenzial - altes pot.

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