import numpy as np
from sb3.stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from sb3.stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs

class VecPBRSWrapperHolistic(VecEnvWrapper):
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
        infos[0]['current_time'] = self.time_step
        current_potentials = _potentials(infos)
        potential_differences = self.gamma * current_potentials - self.last_potentials
        self.last_potentials = current_potentials
        # if infos[0].get("ball_stopped", False):
        #         rewards -= 0.5
        rewards += potential_differences
        self.time_step += 1
        return observations, rewards, dones, infos

def _potential(info, goalie_y_pos = -1.0435):
    ball_pos = info["ball_position"]
    goalie_x_pos = info["goalie_x_position"]
    # Encourage the ball to be closer to the opponent's goal (x=1.34) and goalie to be near the ball
    goal_direction_potential = np.maximum(0, (ball_pos[0] + 1.34) / 2.68)  # Normalize between 0 and 1
    goalie_ball_distance = np.sqrt((goalie_x_pos - ball_pos[0])**2 + (goalie_y_pos - ball_pos[1])**2)
    goalie_ball_potential = np.maximum(0, 1 - goalie_ball_distance / 3)  # Assuming max distance as diagonal of field
    return 0.7 * goal_direction_potential + 0.3 * goalie_ball_potential

def _potentials(infos):
    return np.array([_potential(info) for info in infos])

## Exp Factor
# def _potential(info, goalie_y_pos = -1.0435):
#     ball_pos, goalie_x_pos = info["ball_position"], info["goalie_x_position"]
#     goal_direction_potential = np.exp((ball_pos[0] + 1.34) / 2.68)  # Exponential growth as ball approaches goal
#     goalie_ball_distance = np.sqrt((goalie_x_pos - ball_pos[0])**2 + (goalie_y_pos - ball_pos[1])**2)
#     goalie_ball_potential = np.exp(-goalie_ball_distance)  # Exponential decay with distance
#     return 0.7 * goal_direction_potential + 0.3 * goalie_ball_potential

## Quadratic Cost
# def _potential(info, goalie_y_pos = -1.0435):
#     ball_pos, goalie_x_pos = info["ball_position"], info["goalie_x_position"]
#     goal_direction_potential = ((ball_pos[0] + 1.34) / 2.68) ** 2
#     goalie_ball_distance = np.sqrt((goalie_x_pos - ball_pos[0])**2 + (goalie_y_pos - ball_pos[1])**2)
#     goalie_ball_potential = 1 - (goalie_ball_distance / 1.42) ** 2
#     return 0.7 * goal_direction_potential + 0.3 * goalie_ball_potential

## Asymmetric Costs
# def _potential(ball_pos, goalie_pos):
#     if ball_pos[0] > 0:
#         goal_direction_potential = ((ball_pos[0] + 1.34) / 2.68) ** 2  # Higher weight if moving towards opponent goal
#     else:
#         goal_direction_potential = -((ball_pos[0] + 1.34) / 2.68) ** 2  # Negative or smaller positive weight if moving away
#     goalie_ball_distance = np.sqrt((goalie_pos[0] - ball_pos[0])**2 + (goalie_pos[1] - ball_pos[1])**2)
#     goalie_ball_potential = 1 - (goalie_ball_distance / 1.42) ** 2
#     return 0.7 * goal_direction_potential + 0.3 * goalie_ball_potential

## time based penality (need time for whole or episode scoped time step)
# def _potential(info, goalie_y_pos = -1.0435, max_time = 400):
#     current_time = info["current_time"]
#     ball_pos, goalie_x_pos = info["ball_position"], info["goalie_x_position"]
#     time_factor = current_time / max_time
#     goal_direction_potential = ((ball_pos[0] + 1.34) / 2.68) * time_factor
#     goalie_ball_distance = np.sqrt((goalie_x_pos - ball_pos[0])**2 + (goalie_y_pos - ball_pos[1])**2)
#     goalie_ball_potential = (1 - goalie_ball_distance / 1.42) * (1 - time_factor)
#     return 0.7 * goal_direction_potential + 0.3 * goalie_ball_potential
