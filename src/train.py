import ast
from configparser import ConfigParser
from src.config_logging import save_run_info
from src.sb3.stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, TensorboardCallback

#########
#Custome Added
#############
# copied from https://github.com/DLR-RM/stable-baselines3/issues/1746
from typing import Callable
def linear_schedule(initial_value: float, final_value: float, final_lr_progress: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :param final_value: Final learning rate.
    :param final_lr_progress: The progress at which the final learning rate should be reached.
                              This is a value between 0 and 1, where 1 means 100% of the training.
    :return: A schedule that computes the current learning rate depending on remaining progress.
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining: The remaining progress of the training, decreasing from 1 to 0.
        :return: The current learning rate.
        """
        if (1-progress_remaining) < final_lr_progress:
            # Scale progress to match the final_lr_progress point
            new_lr = initial_value - (initial_value - final_value) * ((1-progress_remaining)/final_lr_progress)
            # scaled_progress = 1 - ((1 - progress_remaining) / final_lr_progress)
            # new_lr = (final_value - initial_value) * scaled_progress + initial_value
            print(f"Learning rate: {new_lr}")
            return new_lr
        else:
            # After reaching the final_lr_progress, keep the learning rate at final_value
            return final_value

    return func

def train_kicker(config: ConfigParser, seed: int, algorithm_class, env):
    alg_config = config['Algorithm']
    try:
        policy_kwargs = ast.literal_eval(alg_config['policy_kwargs'])
        model = algorithm_class(env=env, seed=seed, verbose=1,
                                policy=alg_config['policy'],
                                policy_kwargs=policy_kwargs,
                                tensorboard_log=alg_config['tensorboard_log'],
                                learning_rate=linear_schedule(float(alg_config['learning_rate_start']), float(alg_config['learning_rate_end']), float(alg_config['final_lr_progress']))
                                # Add here more hyperparameters if needed, following the above scheme
                                # alg_config['hyperparameter_name']
                                ################################
                                )
    except KeyError or ValueError:
        # Fall back to default policy_kwargs
        model = algorithm_class(env=env, seed=seed, verbose=1,
                                policy=alg_config['policy'],
                                tensorboard_log=alg_config['tensorboard_log'])

    save_run_info(config=config,
                  seed=seed,
                  algorithm_name=type(model).__name__)

    training_config = config['Training']
    model.learn(total_timesteps=int(training_config['total_timesteps']),
                tb_log_name=training_config['tb_log_name'],
                callback=get_callback(config, seed))
    env.close()


def get_callback(config: ConfigParser, seed: int):
    callback_config = config['Callback']
    checkpoint_callback = CheckpointCallback(name_prefix=f"rl_model_{seed}",
                              save_freq=int(callback_config['save_freq']),
                              save_path=callback_config['save_path'],
                              save_replay_buffer=callback_config.getboolean('save_replay_buffer'),
                              save_vecnormalize=callback_config.getboolean('save_vecnormalize'))
    logging_callback = TensorboardCallback()
    return CallbackList([checkpoint_callback, logging_callback])
