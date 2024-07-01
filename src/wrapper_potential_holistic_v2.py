import numpy as np
from sb3.stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from sb3.stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs

class VecPBRSWrapperHolisticV2(VecEnvWrapper):
    def __init__(self, venv: VecEnv, gamma: float = 0.99):
        super().__init__(venv, venv.observation_space, venv.action_space)
        self.gamma = gamma
        self.last_potentials = None
        self.time_step = 0

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
        self.time_step += 1
        return observations, rewards, dones, infos

def _potential(info: dict):
    # Retrieve necessary components
    ball_pos = info['ball_position']
    ball_vel = info['ball_velocity']
    
    # Weighted potentials considering position towards white goal, away from black goal and staying in-bounds
    white_goal_potential = _white_goal_potential(ball_pos)
    black_goal_risk = _black_goal_risk(ball_pos)
    out_of_bounds_risk = _out_of_bounds_risk(ball_pos)
    vertical_bounds_risk = _vertical_bounds_risk(ball_pos)
    
    # Combine the potentials, with possible tuning of weights
    potential = 2 * white_goal_potential - black_goal_risk - out_of_bounds_risk - vertical_bounds_risk
    return potential

def _potentials(infos: list[dict]):
    return np.array([_potential(info) for info in infos])

def _white_goal_potential(ball_pos):
    # Linear potential increasing as the ball approaches the white goal zone (max x)
    x = ball_pos[0]
    return np.clip((x + 1.34) / 2.68, 0, 1)  # Normalize within [0, 1]

def _black_goal_risk(ball_pos):
    # Linear potential increasing as the ball approaches the black goal zone (min x)
    x = ball_pos[0]
    return np.clip((-x + 1.34) / 2.68, 0, 1)  # Normalize within [0, 1]

def _out_of_bounds_risk(ball_pos):
    # Considering both x and y for out-of-bounds risk
    x, y = ball_pos[0], ball_pos[1]
    x_risk = np.clip(np.abs(x) / 1.34, 0, 1)  # Normalize within [0, 1]
    y_risk = np.clip(np.abs(y) / 0.71, 0, 1)  # Normalize within [0, 1]
    return max(x_risk, y_risk)  # Higher risk taken as potential

def _vertical_bounds_risk(ball_pos):
    # Risk assessment based on the z-position, considering the height of the field
    z = ball_pos[2]
    return np.clip(z / 0.9, 0, 1)  # Normalize within [0, 1], assuming upper bounds are at z=0.9

