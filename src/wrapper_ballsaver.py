import numpy as np

from src.sb3.stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from src.sb3.stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs


class VecBALLSAVERWrapper(VecEnvWrapper):

    def __init__(self, venv: VecEnv, gamma: float = 0.99):
        VecEnvWrapper.__init__(self, venv, venv.observation_space, venv.action_space)
        self.venv = venv
        self.gamma = gamma
        self.previouse_goalie_z_dif = 0

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()
        try:
            _, goalie_z_dif = calculate_intercept(infos[0])
            if goalie_z_dif < self.previouse_goalie_z_dif:
                rewards += 0.1
            #print(goalie_z_dif)
            rewards += (1.573-abs(infos[0]['goalie_angular_pos'])) /10
        except:
            pass
        return observations, rewards, dones, infos

def contour_plot_to_console(X, Z, Y):
    for i in range(len(X)):
        if i % 5 == 0:
            row = ""
            for j in range(len(Z)):
                if j % 5 == 0:
                    row += " " + f"{Y[i][j]:.2f}"
            print(row)


def calculate_intercept(info):
    # Calculate time 't' when ball reaches x = -1.1
    if info["ball_x_velocity"] == 0:
        return "The ball is not moving in the x direction."
    
    t = (-1.0435 - info["ball_x_position"]) / info["ball_x_velocity"]
    
    if t < 0:
        return "The ball has already passed x = -1.0435 or is moving away from it."
    # Calculate z position at time 't'
    z_intercept = info["ball_z_position"] + info["ball_z_velocity"] * t

    goalie_diff = abs(z_intercept - info["goalie_z_position"])  # Difference between z intercept and goalie position

    return t, goalie_diff

'''
        info = {
            "black_conceded": 1 if black_conceded else 0,
            "white_conceded": 1 if white_conceded else 0,
            "ball_position": ball_pos,
            "ball_x_position": ball_pos[0],
            "ball_y_position": ball_pos[1],
            "ball_z_position": ball_pos[2],
            "ball_x_velocity": self.data.qvel[2],
            "ball_z_velocity": self.data.qvel[3],
            "goalie_z_position": self.data.qvel[0],
            "goalie_angular_pos": self.data.qvel[1]
        }
'''