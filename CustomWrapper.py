import numpy as np

from src.sb3.stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from src.sb3.stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs


class CustomWrapper(VecEnvWrapper):

    def __init__(self, venv: VecEnv, gamma: float = 0.99):
        VecEnvWrapper.__init__(self, venv, venv.observation_space, venv.action_space)
        self.venv = venv
        self.gamma = gamma
        # self._print_potential_function_visualization()

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.last_potentials = np.zeros(self.num_envs)
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()
        # Adjust rewards to euclidean distance
        infos = infos[0]
        goalie_pos = np.array([infos["goalie_x_position"], infos["goalie_z_position"]])
        ball_pos = np.array([infos["ball_x_position"], infos["ball_z_position"]])
        euclidean_dist = np.linalg.norm(goalie_pos - ball_pos)
        print(rewards)
        #rewards -= euclidean_dist
        print(rewards)
        return observations, rewards, dones, infos


def contour_plot_to_console(X, Z, Y):
    for i in range(len(X)):
        if i % 5 == 0:
            row = ""
            for j in range(len(Z)):
                if j % 5 == 0:
                    row += " " + f"{Y[i][j]:.2f}"
            print(row)
