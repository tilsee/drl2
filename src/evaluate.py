def load_model(model_path: str, algorithm_class, env):
    model = algorithm_class.load(model_path)
    model.set_env(env)

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)
        if dones[0]:
            obs = env.reset()
    env.close()
