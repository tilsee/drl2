import json
import os
def load_model(model_path: str, algorithm_class, env):
    model = algorithm_class.load(model_path)
    model.set_env(env)

    obs = env.reset()
    white_goals = 0
    black_goals = 0
    outside_bounds = 0
    dones_count = 0 
    for _ in range(10000):
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
