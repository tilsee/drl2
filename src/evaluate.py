import json
import os
from configparser import ConfigParser
from pathlib import Path

from environment import load_normalized_kicker_env, create_kicker_env
from sb3.stable_baselines3.common.evaluation import evaluate_policy

def load_model(model_path: str, algorithm_class, env):
    model = algorithm_class.load(model_path)
    model.set_env(env)

    obs = env.reset()
    white_goals = 0
    black_goals = 0
    outside_bounds = 0
    dones_count = 0 
    for _ in range(5000):
        action, _states = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(action)
        white_goals += infos[0]['white_conceded']
        black_goals += infos[0]['black_conceded']
        outside_bounds += infos[0]['outside_bounds']
        if dones[0]:
            dones_count += 1
            obs = env.reset()
    # Construct the results dictionary
    results = {
        "white_goals": white_goals,
        "black_goals": black_goals,
        "outside_bounds": outside_bounds,
        "timeouts": dones_count-white_goals-black_goals-outside_bounds
    }
    model_dir = str(model_path).replace(".zip", "evaluation_results.json")
    
    # Construct the path for the new JSON file
    results_path =model_dir
    
    # Write the dictionary to the file
    with open(results_path, 'w') as f:
        json.dump(results, f)
    env.close()


def evaluate_model(config: ConfigParser, save_path: Path, algorithm_class, normalize_env_path: str = None):
    model = algorithm_class.load(config['Testing']['test_model_path'])
    if normalize_env_path is None:
        env = create_kicker_env(seed=config['Testing'].getint('eval_seed'), config=config)
    else:
        env = load_normalized_kicker_env(config=config, seed=config['Testing'].getint('eval_seed'),
                                         normalize_path=config['Testing']['normalized_env_path'])
    episode_rewards, episode_lengths = evaluate_policy(model=model, env=env,
                                                       n_eval_episodes=config['Testing'].getint('num_eval_episodes'))
    save_results(config=config, save_path=save_path,
                 episode_rewards=episode_rewards, episode_lengths=episode_lengths)

    print("-" * 50)
    print(f"Mean reward: {episode_rewards}, Mean episode length: {episode_lengths}")
    print("-" * 50)


def save_results(config: ConfigParser, save_path: Path, episode_rewards, episode_lengths):
    with open(save_path / f'evaluation_result_{round(time.time() * 1000)}.txt', 'w') as f:
        f.write(f"Experiment name: {config['Common']['experiment_name']}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Seed: {config['Testing'].getint('eval_seed')}\n")
        f.write(f"Number of evaluation episodes: {config['Testing'].getint('num_eval_episodes')}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Mean reward: {episode_rewards}\n")
        f.write(f"Mean episode length: {episode_lengths}\n")
        f.write("-" * 50 + "\n")
