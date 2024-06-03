from configparser import ConfigParser, ExtendedInterpolation

from src.environment import create_kicker_env
from src.sb3.stable_baselines3 import A2C
from src.tensorboard_aggregator import aggregator
from train import train_kicker


def main():
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read('../resources/config.ini')
    used_rl_algorithm = A2C
    for seed in range(1, 4):
        env = create_kicker_env(config=config, seed=seed)
        train_kicker(config=config, seed=seed, algorithm_class=used_rl_algorithm, env=env)
    aggregator.main(path_arg=config['Algorithm']['tensorboard_log'])


if __name__ == '__main__':
    main()
