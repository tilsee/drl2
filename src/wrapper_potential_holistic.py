import numpy as np
from sb3.stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from sb3.stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs

class VecPBRSWrapperHolistic(VecEnvWrapper):
    def __init__(self, venv: VecEnv, gamma: float = 0.99):
        super().__init__(venv, venv.observation_space, venv.action_space)
        self.gamma = gamma
        self.last_potentials = None

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.last_potentials = np.zeros(self.num_envs)
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()
        current_potentials = _potentials(infos)
        potential_differences = self.gamma * current_potentials - self.last_potentials
        self.last_potentials = current_potentials
        rewards += potential_differences
        return observations, rewards, dones, infos

def _potential(ball_pos, goalie_pos):
    # Encourage the ball to be closer to the opponent's goal (x=1.34) and goalie to be near the ball
    goal_direction_potential = np.maximum(0, (ball_pos[0] + 1.34) / 2.68)  # Normalize between 0 and 1
    goalie_ball_distance = np.sqrt((goalie_pos[0] - ball_pos[0])**2 + (goalie_pos[1] - ball_pos[1])**2)
    goalie_ball_potential = np.maximum(0, 1 - goalie_ball_distance / 1.42)  # Assuming max distance as diagonal of field
    return 0.7 * goal_direction_potential + 0.3 * goalie_ball_potential

def _potentials(infos):
    return np.array([_potential(info["ball_position"], info.get("goalie_position", [0, 0])) for info in infos])

def contour_plot_to_console(X, Z, Y):
    for i in range(len(X)):
        if i % 5 == 0:
            row = ""
            for j in range(len(Z)):
                if j % 5 == 0:
                    row += " " + f"{Y[i][j]:.2f}"
            print(row)
