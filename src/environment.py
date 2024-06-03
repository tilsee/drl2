from configparser import ConfigParser

from kicker.kicker_env import Kicker
from sb3.stable_baselines3.common.monitor import Monitor
from sb3.stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from sb3.stable_baselines3.common.vec_env.vec_pbrs import VecPBRSWrapper
from sb3.stable_baselines3.common.vec_env.vec_normalize import VecNormalize


def create_kicker_env(config: ConfigParser, seed: int):
    env_conf = config['Kicker']
    env = Kicker(seed=seed,
                 horizon=int(env_conf['horizon']),
                 continuous_act_space=env_conf.getboolean('continuous_act_space'),
                 multi_discrete_act_space=env_conf.getboolean('multi_discrete_act_space'),
                 image_obs_space=env_conf.getboolean('image_obs_space'),
                 end_episode_on_struck_goal=env_conf.getboolean('end_episode_on_struck_goal'),
                 end_episode_on_conceded_goal=env_conf.getboolean('end_episode_on_conceded_goal'),
                 reset_goalie_position=env_conf.getboolean('reset_goalie_position'),
                 render_training=env_conf.getboolean('render_training'),
                 lateral_bins=env_conf.getint('lateral_bins'),
                 angular_bins=env_conf.getint('angular_bins'),
                 step_frequency=env_conf.getint('step_frequency'))
    # Default wrappers
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    ############################################
    # Add Wrappers here
    ############################################

    if not env_conf.getboolean('render_training'):
        video_conf = config['VideoRecording']
        print(f"Recording video every {video_conf.getint('video_interval')} steps with a length of "
              f"{video_conf.getint('video_length')} frames, saving to {video_conf['video_folder']}")
        env = VecVideoRecorder(venv=env, name_prefix=f"rl-kicker-video-{seed}",
                               record_video_trigger=lambda x: x % video_conf.getint('video_interval') == 0,
                               video_length=video_conf.getint('video_length'),
                               video_folder=video_conf['video_folder'])
    env.seed(seed)
    return env


def load_normalized_kicker_env(config: ConfigParser, seed: int, normalize_path: str):
    env = create_kicker_env(seed=seed, config=config)
    env = VecNormalize.load(normalize_path, env)
    return env
