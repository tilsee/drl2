import numpy as np
from sb3.stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from sb3.stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs

class VecPBRSWrapper2(VecEnvWrapper):

    def __init__(self, venv: VecEnv, gamma: float = 0.99):
        super().__init__(venv)
        self.gamma = gamma
        self.last_potentials = None

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.last_potentials = np.zeros(self.num_envs)
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()
        current_potentials = self._potentials(infos)
        potential_differences = self.gamma * current_potentials - self.last_potentials
        self.last_potentials = current_potentials
        rewards += potential_differences
        return observations, rewards, dones, infos

    def _potentials(self, infos: list[dict]):
        return np.array([self._potential(info["ball_position"]) for info in infos])

    def _potential(self, ball_pos: np.ndarray):
        # Define the potential function based on ball position
        x, y = ball_pos[0], ball_pos[1]
        goal_x = -1.25  # Assuming the goal is at x = -1.25
        distance_to_goal = np.sqrt((x - goal_x) ** 2 + y ** 2)
        return -distance_to_goal  # Negative distance as the potential (closer to goal is higher potential)

def contour_plot_to_console(X, Z, Y):
    for i in range(len(X)):
        if i % 5 == 0:
            row = ""
            for j in range(len(Z)):
                if j % 5 == 0:
                    row += " " + f"{Y[i][j]:.2f}"
            print(row)
