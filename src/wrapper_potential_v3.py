import numpy as np
from sb3.stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from sb3.stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs

class VecPBRSWrapperV3(VecEnvWrapper):
    def __init__(self, venv: VecEnv, gamma: float = 0.99, action_smoothness_penalty: float = 0.05):
        super().__init__(venv)
        self.gamma = gamma
        self.last_potentials = None
        self.last_actions = None
        self.action_smoothness_penalty = action_smoothness_penalty

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.last_potentials = np.zeros(self.num_envs)
        # Initialize last_actions based on the action space type and shape
        if hasattr(self.venv.action_space, 'shape'):
            self.last_actions = np.zeros((self.num_envs, *self.venv.action_space.shape))
        else:  # Handle discrete action spaces
            self.last_actions = np.zeros((self.num_envs, 1))
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()
        current_potentials = self._potentials(infos)
        potential_differences = self.gamma * current_potentials - self.last_potentials
        self.last_potentials = current_potentials

        # Retrieve the current actions from infos if available, otherwise assume zero change
        current_actions = np.array([info.get('action', np.zeros_like(self.last_actions[i]))
                                    for i, info in enumerate(infos)])

        # Calculate the difference in actions to apply the smoothness penalty
        action_differences = np.linalg.norm(current_actions - self.last_actions, axis=1)
        rewards -= self.action_smoothness_penalty * action_differences

        # Update last actions to the current actions
        self.last_actions = current_actions

        # Add the potential differences to the rewards
        rewards += potential_differences
        return observations, rewards, dones, infos

    def _potentials(self, infos: list[dict]):
        return np.array([self._potential(info) for info in infos])

    def _potential(self, info: dict):
        # Calculate potential based on the ball's position
        ball_pos = info['ball_position']
        x, y = ball_pos[0], ball_pos[1]
        white_goal_x = 1.216  # Approximate x-coordinate of the white goal
        black_goal_x = -1.216  # Approximate x-coordinate of the black goal

        # Distance to both goals
        distance_to_white_goal = np.sqrt((x - white_goal_x) ** 2 + y ** 2)
        distance_to_black_goal = np.sqrt((x - black_goal_x) ** 2 + y ** 2)

        # Inverse of distance for higher potential when closer to goal
        potential_white = 1 / (distance_to_white_goal + 1e-3)  # small epsilon to avoid division by zero
        potential_black = 1 / (distance_to_black_goal + 1e-3)

        # Potential is higher when closer to white goal and further from black goal
        return potential_white - potential_black
