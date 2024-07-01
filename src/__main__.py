import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from evaluate import load_model
import glob
import numpy as np

from configparser import ConfigParser, ExtendedInterpolation

from src.environment import create_kicker_env, load_normalized_kicker_env
from src.sb3.stable_baselines3 import PPO #, A2C
from src.tensorboard_aggregator import aggregator
from train import train_kicker


def main():
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read('resources/config.ini')
    used_rl_algorithm = PPO #,A2C
    for seed in [1]:#range(1, 4):
        env = create_kicker_env(config=config, seed=seed)
        train_kicker(config=config, seed=seed, algorithm_class=used_rl_algorithm, env=env)
    aggregator.main(path_arg=config['Algorithm']['tensorboard_log'])
    model_files = glob.glob(os.path.join(config["Callback"]["save_path"], f"*{config['Training']['total_timesteps']}_steps.zip"))
    vecnorml_files = glob.glob(os.path.join(config["Callback"]["save_path"], f"*{config['Training']['total_timesteps']}_steps.pkl"))
    model_files.sort()
    vecnorml_files.sort()
    for i, model_file in enumerate(model_files):
        print(10*"*")
        print(f"Loading model {model_file}")
        print(f"Loading vecnorm {vecnorml_files[i]}")
        print(f"Seed {i+1}")
        print(10*"*")
        env = load_normalized_kicker_env(config=config, seed=i+1, override_video=True, normalize_path=vecnorml_files[i])
        load_model(model_file, used_rl_algorithm, env)

if __name__ == '__main__':
    main()
